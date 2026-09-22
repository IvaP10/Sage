using System.Buffers.Binary;
using System.IO.Pipes;
using System.Security.Cryptography;
using System.Text;
using Google.Protobuf;
using Sage.Ipc.V2;

namespace Sage.Windows;

internal sealed class SageCoreClient : IDisposable
{
    private string _authenticatedSessionId = "";
    private const int ProtocolVersion = 2;
    private const int MaximumFrameBytes = 4 * 1024 * 1024;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly CancellationTokenSource _stopping = new();
    private NamedPipeClientStream? _pipe;
    private long _sequence;

    public event EventHandler<CoreEvent>? EventReceived;
    public Func<AdapterRequest, Task<AdapterResult>>? AdapterHandler { get; set; }

    public async Task ConnectAsync()
    {
        var pipe = new NamedPipeClientStream(
            ".",
            "sage-core-v2",
            PipeDirection.InOut,
            PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly
        );
        await pipe.ConnectAsync(1_000, _stopping.Token);
        _pipe = pipe;
        await AuthenticateAsync(pipe, _stopping.Token);
        _ = Task.Run(() => ReceiveLoopAsync(pipe, _stopping.Token));
        await WriteFrameAsync(pipe, new Frame { ProtocolVersion = ProtocolVersion, Sequence = NextSequence(), AdapterHello = new AdapterHello { Domain = "native" } }, _stopping.Token);
    }

    public Task SubmitTaskAsync(string text, string conversationId = "", string? folder = null)
    {
        var submit = new SubmitTask { Text = text, Source = InputSource.Typed, ConversationId = conversationId };
        if (!string.IsNullOrEmpty(folder)) { var scope = new ResourceScope { Root = folder }; scope.Effects.Add(ResourceEffect.Read); submit.Resources.Add(scope); }
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            SubmitTask = submit,
        });
    }

    public Task UnlockStorageAsync() => SendCommandAsync(new UiCommand { RequestId = Guid.NewGuid().ToString(), UnlockStorage = new UnlockStorage() });

    public Task ControlTaskAsync(string taskId, ControlTask.Types.Operation operation)
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            ControlTask = new ControlTask
            {
                TaskId = taskId,
                Operation = operation,
            },
        });
    }

    public Task UndoAsync(string taskId)
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            UndoLastAction = new UndoLastAction { TaskId = taskId },
        });
    }

    public Task RequestStateAsync(bool includeCompleted)
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            GetState = new GetState { IncludeCompletedTasks = includeCompleted },
        });
    }

    public Task SaveProviderSettingsAsync(
        string provider,
        string model,
        string endpoint,
        string apiKey,
        bool removeSavedKey,
        bool nativeAuthenticationSatisfied
    )
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            SaveProviderSettings = new SaveProviderSettings
            {
                Role = "reasoning",
                Provider = provider,
                Model = model,
                Endpoint = endpoint,
                ApiKey = apiKey,
                RemoveSavedKey = removeSavedKey,
                NativeAuthenticationSatisfied = nativeAuthenticationSatisfied,
            },
        });
    }

    public Task TestProviderConnectionAsync(
        string provider,
        string model,
        string endpoint,
        string apiKey
    )
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            TestProviderConnection = new TestProviderConnection
            {
                Role = "reasoning",
                Provider = provider,
                Model = model,
                Endpoint = endpoint,
                ApiKey = apiKey,
            },
        });
    }

    public Task ResolveApprovalAsync(
        ApprovalRequest approval,
        bool approve,
        bool nativeAuthenticationSatisfied
    )
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            ApprovalResponse = new ApprovalResponse
            {
                TaskId = approval.TaskId,
                ActionId = approval.ActionId,
                ApprovalId = approval.ApprovalId,
                ApprovalDigest = approval.ApprovalDigest,
                Decision = approve ? ApprovalDecision.ApproveOnce : ApprovalDecision.Deny,
                NativeAuthenticationSatisfied = nativeAuthenticationSatisfied,
            },
        });
    }

    public Task AnswerAsync(QuestionRequest question, string answer)
    {
        return SendCommandAsync(new UiCommand
        {
            RequestId = Guid.NewGuid().ToString(),
            UserAnswer = new UserAnswer
            {
                TaskId = question.TaskId,
                ActionId = question.ActionId,
                QuestionId = question.QuestionId,
                Answer = answer,
            },
        });
    }

    private async Task AuthenticateAsync(NamedPipeClientStream pipe, CancellationToken cancellationToken)
    {
        var challengeFrame = await ReadFrameAsync(pipe, cancellationToken);
        if (challengeFrame.ProtocolVersion != ProtocolVersion
            || challengeFrame.PayloadCase != Frame.PayloadOneofCase.ServerChallenge
            || challengeFrame.ServerChallenge.Nonce.Length != 32)
        {
            throw new InvalidDataException("SAGE Core sent an invalid authentication challenge");
        }
        var secret = IpcSecretStore.LoadOrCreate();
        var nonce = RandomNumberGenerator.GetBytes(32);
        var version = typeof(SageCoreClient).Assembly.GetName().Version?.ToString(3) ?? "1.0.0";
        var authentication = new ClientAuthenticate
        {
            ClientKind = ClientKind.Windows,
            ClientVersion = version,
            ClientNonce = ByteString.CopyFrom(nonce),
            Proof = ByteString.CopyFrom(CreateProof(
                secret,
                challengeFrame.ServerChallenge.Nonce.ToByteArray(),
                nonce,
                ProtocolVersion,
                (int)ClientKind.Windows,
                version
            )),
        };
        await WriteFrameAsync(pipe, new Frame
        {
            ProtocolVersion = ProtocolVersion,
            Sequence = NextSequence(),
            ClientAuthenticate = authentication,
        }, cancellationToken);
        var result = await ReadFrameAsync(pipe, cancellationToken);
        if (result.PayloadCase != Frame.PayloadOneofCase.AuthenticationResult
            || !result.AuthenticationResult.Accepted)
        {
            throw new UnauthorizedAccessException("SAGE Core rejected local IPC authentication");
        }
        using var serverMessage = new MemoryStream();
        serverMessage.Write(Encoding.UTF8.GetBytes("SAGE-CORE-PROOF-V2\0"));
        serverMessage.Write(authentication.Proof.ToByteArray());
        var length = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(length, (uint)Encoding.UTF8.GetByteCount(result.AuthenticationResult.SessionId));
        serverMessage.Write(length);
        serverMessage.Write(Encoding.UTF8.GetBytes(result.AuthenticationResult.SessionId));
        var expected = HMACSHA256.HashData(secret, serverMessage.ToArray());
        if (!CryptographicOperations.FixedTimeEquals(expected, result.AuthenticationResult.ServerProof.ToByteArray()))
            throw new UnauthorizedAccessException("Core identity proof is invalid");
        _authenticatedSessionId = result.AuthenticationResult.SessionId;
    }

    private async Task SendCommandAsync(UiCommand command)
    {
        var pipe = _pipe ?? throw new InvalidOperationException("SAGE Core is not connected");
        await WriteFrameAsync(pipe, new Frame
        {
            ProtocolVersion = ProtocolVersion,
            Sequence = NextSequence(),
            UiCommand = command,
        }, _stopping.Token);
    }

    public Task KnowledgeAsync(KnowledgeCommand command) => SendCommandAsync(new UiCommand { RequestId = Guid.NewGuid().ToString(), KnowledgeCommand = command });
    public Task WorkflowAsync(WorkflowCommand command) => SendCommandAsync(new UiCommand { RequestId = Guid.NewGuid().ToString(), WorkflowCommand = command });

    private async Task ReceiveLoopAsync(Stream pipe, CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var frame = await ReadFrameAsync(pipe, cancellationToken);
                if (frame.PayloadCase == Frame.PayloadOneofCase.CoreEvent)
                {
                    EventReceived?.Invoke(this, frame.CoreEvent);
                }
                if (frame.PayloadCase == Frame.PayloadOneofCase.AdapterRequest && AdapterHandler is not null)
                {
                    if (frame.AdapterRequest.Operation == "execute" && frame.AdapterRequest.Grant?.WorkerSession != _authenticatedSessionId)
                        throw new UnauthorizedAccessException("Grant belongs to another worker session");
                    var response = await AdapterHandler(frame.AdapterRequest);
                    await WriteFrameAsync(pipe, new Frame { ProtocolVersion = ProtocolVersion, Sequence = NextSequence(), AdapterResult = response }, cancellationToken);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (IOException) when (cancellationToken.IsCancellationRequested) { }
    }

    private async Task WriteFrameAsync(Stream stream, Frame frame, CancellationToken cancellationToken)
    {
        var payload = frame.ToByteArray();
        if (payload.Length == 0 || payload.Length > MaximumFrameBytes)
        {
            throw new InvalidDataException("IPC frame is outside the accepted size range");
        }
        var header = new byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(header, (uint)payload.Length);
        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            await stream.WriteAsync(header, cancellationToken);
            await stream.WriteAsync(payload, cancellationToken);
            await stream.FlushAsync(cancellationToken);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private static async Task<Frame> ReadFrameAsync(Stream stream, CancellationToken cancellationToken)
    {
        var header = new byte[4];
        await stream.ReadExactlyAsync(header, cancellationToken);
        var length = BinaryPrimitives.ReadUInt32BigEndian(header);
        if (length == 0 || length > MaximumFrameBytes)
        {
            throw new InvalidDataException("SAGE Core sent an invalid frame length");
        }
        var payload = new byte[checked((int)length)];
        await stream.ReadExactlyAsync(payload, cancellationToken);
        return Frame.Parser.ParseFrom(payload);
    }

    private static byte[] CreateProof(
        byte[] secret,
        byte[] serverNonce,
        byte[] clientNonce,
        uint protocolVersion,
        int clientKind,
        string clientVersion
    )
    {
        using var stream = new MemoryStream();
        stream.Write(Encoding.UTF8.GetBytes("SAGE-LOCAL-IPC-AUTH-V2\0"));
        stream.Write(serverNonce);
        stream.Write(clientNonce);
        Span<byte> integer = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(integer, protocolVersion);
        stream.Write(integer);
        BinaryPrimitives.WriteInt32BigEndian(integer, clientKind);
        stream.Write(integer);
        var version = Encoding.UTF8.GetBytes(clientVersion);
        BinaryPrimitives.WriteUInt32BigEndian(integer, (uint)version.Length);
        stream.Write(integer);
        stream.Write(version);
        return HMACSHA256.HashData(secret, stream.ToArray());
    }

    private ulong NextSequence() => unchecked((ulong)Interlocked.Increment(ref _sequence));

    public void Dispose()
    {
        _stopping.Cancel();
        _pipe?.Dispose();
        _writeLock.Dispose();
        _stopping.Dispose();
    }
}
