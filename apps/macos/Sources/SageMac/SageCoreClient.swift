import CryptoKit
import Foundation
import Network
import SwiftProtobuf

final class SageCoreClient: @unchecked Sendable {
    var onEvent: (@Sendable (Sage_Ipc_V2_CoreEvent) -> Void)?
    var onDisconnect: (@Sendable (String) -> Void)?
    var onAdapterRequest: (@Sendable (Sage_Ipc_V2_AdapterRequest) async -> Sage_Ipc_V2_AdapterResult)?

    private let queue = DispatchQueue(label: "com.ivanpadeliya.sage.ipc")
    private let stateLock = NSLock()
    private var connection: NWConnection?
    private var negotiatedFeatures: Set<String> = []
    private var authenticatedSessionID = ""
    private var adapterOperations = AdapterOperations()
    private let writer = OrderedFrameWriter()
    private let submissions = PendingSubmissions()

    func connect() async throws {
        disconnect()
        let socket = try Self.socketPath()
        let connection = NWConnection(to: .unix(path: socket), using: .tcp)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let gate = ContinuationGate(continuation)
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    gate.resume()
                case .failed(let error):
                    gate.resume(throwing: SageClientError.connectionFailed(error.localizedDescription))
                case .cancelled:
                    gate.resume(throwing: SageClientError.connectionFailed("IPC connection cancelled"))
                default:
                    break
                }
            }
            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + .seconds(1)) {
                if gate.resume(
                    throwing: SageClientError.connectionFailed(
                        "SAGE Core did not open its Unix socket within one second"
                    )
                ) {
                    connection.cancel()
                }
            }
        }
        let operations = AdapterOperations()
        stateLock.withLock { self.connection = connection; self.adapterOperations = operations }
        do { try await authenticate(connection) }
        catch { disconnect(); throw error }
        receiveLoop(connection, operations: operations)
        var hello = Sage_Ipc_V2_AdapterHello()
        hello.domain = "native"
        hello.cancellationProtocol = 1
        try await sendPayload(.adapterHello(hello))
        for entry in submissions.snapshot() { try await sendSubmission(entry) }
    }

    func disconnect() {
        let prior = stateLock.withLock {
            let prior = (connection, adapterOperations)
            connection = nil
            authenticatedSessionID = ""
            negotiatedFeatures = []
            return prior
        }
        prior.0?.cancel()
        prior.1.close()
    }

    func submitTask(
        _ text: String,
        source: Sage_Ipc_V2_InputSource = .typed,
        conversationID: String = "",
        folders: [String] = [],
        supersedesTaskID: String = "",
        voiceStreamID: String = "",
        streamedPrefix: Bool = false,
        finalizeStream: Bool = false,
        requestID: String? = nil,
        onRequestIDStaged: (@MainActor @Sendable (String) -> Void)? = nil
    ) async throws {
        var submit = Sage_Ipc_V2_SubmitTask()
        submit.text = text
        submit.source = source
        submit.conversationID = conversationID
        submit.supersedesTaskID = supersedesTaskID
        submit.voiceStreamID = voiceStreamID
        submit.streamedPrefix = streamedPrefix
        submit.finalizeStream = finalizeStream
        submit.resources = folders.map { path in var scope = Sage_Ipc_V2_ResourceScope(); scope.root = path; scope.effects = [.read]; return scope }
        let entry = try submissions.stage(submit.serializedData(), requestID: requestID)
        await onRequestIDStaged?(entry.requestID)
        try await sendSubmission(entry)
    }

    private func sendSubmission(_ entry: PendingSubmissions.Entry) async throws {
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = entry.requestID
        command.command = .submitTask(try Sage_Ipc_V2_SubmitTask(serializedBytes: entry.payload))
        try await send(command)
    }

    func unlockStorage() async throws {
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .unlockStorage(Sage_Ipc_V2_UnlockStorage())
        try await send(command)
    }

    func updateIntent(streamID: String, revision: UInt64, text: String, activeTaskID: String, allowInterrupt: Bool, voiceInput: Bool = false, folders: [String]) async throws {
        var input = Sage_Ipc_V2_UpdateIntent()
        input.streamID = streamID
        input.revision = revision
        input.text = text
        input.activeTaskID = activeTaskID
        input.allowInterrupt = allowInterrupt
        input.voiceInput = voiceInput
        input.folderRoots = folders
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .updateIntent(input)
        try await send(command)
    }

    func requestState(includeCompleted: Bool) async throws {
        var request = Sage_Ipc_V2_GetState()
        request.includeCompletedTasks = includeCompleted
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .getState(request)
        try await send(command)
    }

    func resolveApproval(
        _ approval: Sage_Ipc_V2_ApprovalRequest,
        approve: Bool,
        nativeAuthenticationSatisfied: Bool
    ) async throws {
        var response = Sage_Ipc_V2_ApprovalResponse()
        response.taskID = approval.taskID
        response.actionID = approval.actionID
        response.approvalID = approval.approvalID
        response.approvalDigest = approval.approvalDigest
        response.decision = approve ? .approveOnce : .deny
        response.nativeAuthenticationSatisfied = nativeAuthenticationSatisfied
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .approvalResponse(response)
        try await send(command)
    }

    func answer(_ question: Sage_Ipc_V2_QuestionRequest, text: String) async throws {
        var answer = Sage_Ipc_V2_UserAnswer()
        answer.taskID = question.taskID
        answer.actionID = question.actionID
        answer.questionID = question.questionID
        answer.answer = text
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .userAnswer(answer)
        try await send(command)
    }

    func control(taskID: String, operation: Sage_Ipc_V2_ControlTask.Operation) async throws {
        var control = Sage_Ipc_V2_ControlTask()
        control.taskID = taskID
        control.operation = operation
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .controlTask(control)
        try await send(command)
    }

    func undo(taskID: String, actionID: String) async throws {
        var undo = Sage_Ipc_V2_UndoLastAction()
        undo.taskID = taskID
        undo.actionID = actionID
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .undoLastAction(undo)
        try await send(command)
    }

    private func authenticate(_ connection: NWConnection) async throws {
        let challengeFrame = try await receiveFrame(connection)
        guard challengeFrame.protocolVersion == 2,
              case .serverChallenge(let challenge) = challengeFrame.payload,
              challenge.nonce.count == 32 else {
            throw SageClientError.authenticationFailed("SAGE Core sent an invalid challenge")
        }
        let secret = try IPCSecretStore().loadOrCreateSecret()
        var clientNonce = Data(count: 32)
        let randomStatus = clientNonce.withUnsafeMutableBytes { buffer in
            SecRandomCopyBytes(kSecRandomDefault, 32, buffer.baseAddress!)
        }
        guard randomStatus == errSecSuccess else {
            throw SageClientError.authenticationFailed("Could not generate client nonce")
        }
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"
        let clientFeatures = challenge.supportedFeatures
            .filter { ["world_model_v1", "application_control_v1", "procedure_execution_v1", "native_file_stream_v1", "lan_device_discovery_v1", "lan_renderer_observation_v1"].contains($0) }
            .sorted()
        var authentication = Sage_Ipc_V2_ClientAuthenticate()
        authentication.clientKind = .macos
        authentication.clientVersion = version
        authentication.clientNonce = clientNonce
        authentication.supportedFeatures = clientFeatures
        authentication.proof = Self.proof(
            secret: secret,
            serverNonce: challenge.nonce,
            clientNonce: clientNonce,
            protocolVersion: 2,
            clientKind: Sage_Ipc_V2_ClientKind.macos.rawValue,
            clientVersion: version,
            features: clientFeatures
        )
        var frame = Sage_Ipc_V2_Frame()
        frame.protocolVersion = 2
        frame.payload = .clientAuthenticate(authentication)
        try await writeFrame(frame, connection: connection)
        let resultFrame = try await receiveFrame(connection)
        guard resultFrame.protocolVersion == 2,
              case .authenticationResult(let result) = resultFrame.payload, result.accepted else {
            throw SageClientError.authenticationFailed("SAGE Core rejected local IPC authentication")
        }
        var serverMessage = Data("SAGE-CORE-PROOF-V2\0".utf8)
        serverMessage.append(authentication.proof)
        var length = UInt32(result.sessionID.utf8.count).bigEndian
        withUnsafeBytes(of: &length) { serverMessage.append(contentsOf: $0) }
        serverMessage.append(Data(result.sessionID.utf8))
        let negotiated = result.negotiatedFeatures.sorted()
        guard Set(negotiated).count == negotiated.count,
              Set(negotiated).isSubset(of: Set(clientFeatures)) else {
            throw SageClientError.authenticationFailed("SAGE Core negotiated an unsupported protocol feature")
        }
        Self.appendFeatureTranscript(negotiated, to: &serverMessage)
        guard HMAC<SHA256>.isValidAuthenticationCode(result.serverProof, authenticating: serverMessage, using: SymmetricKey(data: secret)) else {
            throw SageClientError.authenticationFailed("Core identity proof is invalid")
        }
        stateLock.withLock {
            authenticatedSessionID = result.sessionID
            negotiatedFeatures = Set(negotiated)
        }
    }

    private func send(_ command: Sage_Ipc_V2_UiCommand) async throws {
        try await sendPayload(.uiCommand(command))
    }

    func knowledge(_ request: Sage_Ipc_V2_KnowledgeCommand) async throws {
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .knowledgeCommand(request)
        try await send(command)
    }

    func workflow(_ request: Sage_Ipc_V2_WorkflowCommand) async throws {
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .workflowCommand(request)
        try await send(command)
    }

    func worldModel(_ request: Sage_Ipc_V2_WorldModelCommand) async throws {
        guard stateLock.withLock({ negotiatedFeatures.contains("world_model_v1") }) else {
            throw SageClientError.authenticationFailed(
                "This Sage Core connection does not support world-model commands"
            )
        }
        if ["run_goal", "run_controller", "run_stream_procedure", "run_file_stream_copy"].contains(request.operation),
           !stateLock.withLock({ negotiatedFeatures.contains("procedure_execution_v1") }) {
            throw SageClientError.authenticationFailed(
                "This Sage Core connection does not support procedure execution"
            )
        }
        if ["run_stream_procedure", "run_file_stream_copy"].contains(request.operation),
           !stateLock.withLock({ negotiatedFeatures.contains("native_file_stream_v1") }) {
            throw SageClientError.authenticationFailed(
                "This Sage Core connection does not support native file streaming"
            )
        }
        if request.operation == "discover_upnp_media_renderers",
           !stateLock.withLock({ negotiatedFeatures.contains("lan_device_discovery_v1") }) {
            throw SageClientError.authenticationFailed(
                "This Sage Core connection does not support private-LAN device discovery"
            )
        }
        if ["observe_upnp_transport", "observe_upnp_protocol_info"].contains(request.operation),
           !stateLock.withLock({
               negotiatedFeatures.contains("lan_device_discovery_v1")
                   && negotiatedFeatures.contains("lan_renderer_observation_v1")
           }) {
            throw SageClientError.authenticationFailed(
                "This Sage Core connection does not support renderer-state observations"
            )
        }
        var command = Sage_Ipc_V2_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .worldModelCommand(request)
        try await send(command)
    }

    private func sendPayload(_ payload: Sage_Ipc_V2_Frame.OneOf_Payload) async throws {
        guard let connection = stateLock.withLock({ self.connection }) else {
            throw SageClientError.connectionFailed("SAGE Core is not connected")
        }
        var frame = Sage_Ipc_V2_Frame()
        frame.protocolVersion = 2
        frame.payload = payload
        try await writeFrame(frame, connection: connection)
    }

    private func receiveLoop(_ connection: NWConnection, operations: AdapterOperations) {
        Task.detached { [weak self] in
            do {
                while !Task.isCancelled {
                    let frame = try await self?.receiveFrame(connection)
                    guard let frame else { return }
                    if case .coreEvent(let event) = frame.payload {
                        if case .taskAccepted(let receipt) = event.event {
                            self?.submissions.resolve(receipt.requestID)
                        } else if case .error(let error) = event.event, error.code != "ipc_command_retryable" {
                            self?.submissions.resolve(error.requestID)
                        }
                        self?.onEvent?(event)
                    }
                    if case .adapterRequest(let request) = frame.payload, let handler = self?.onAdapterRequest {
                        guard let self else { return }
                        let authenticatedSession = self.stateLock.withLock { self.authenticatedSessionID }
                        if request.operation == "execute", request.grant.workerSession != authenticatedSession {
                            throw SageClientError.authenticationFailed("Grant belongs to another session")
                        }
                        if request.operation == "probe_control" {
                            let payload = try JSONSerialization.jsonObject(with: Data(request.json.utf8)) as? [String: Any]
                            let lease = payload?["probe_lease"] as? [String: Any]
                            guard let lease,
                                  UUID(uuidString: lease["id"] as? String ?? "") != nil,
                                  lease["worker_session"] as? String == authenticatedSession else {
                                throw SageClientError.authenticationFailed("Learning probe belongs to another authenticated adapter session")
                            }
                        }
                        do {
                            let started = try operations.start(id: request.requestID, expiresAt: request.expiresAtUnixMs) { [weak self] in
                                let response = await handler(request)
                                do { try await self?.sendAdapterPayload(.adapterResult(response), connection: connection) }
                                catch { self?.failSession(connection, operations: operations, message: error.localizedDescription) }
                            }
                            if !started { try await self.sendAdapterFailure(request.requestID, "Stopped before worker admission", connection: connection) }
                        } catch AdapterOperations.Failure.capacity {
                            try await self.sendAdapterFailure(request.requestID, "Worker capacity reached before execution", connection: connection)
                        }
                    }
                    if case .adapterCancel(let cancellation) = frame.payload {
                        try operations.cancel(id: cancellation.requestID, expiresAt: cancellation.expiresAtUnixMs)
                        var acknowledgement = Sage_Ipc_V2_AdapterCancelAcknowledged()
                        acknowledgement.requestID = cancellation.requestID
                        try await self?.sendAdapterPayload(.adapterCancelAcknowledged(acknowledgement), connection: connection)
                    }
                }
            } catch {
                guard let self else { return }
                self.failSession(connection, operations: operations, message: error.localizedDescription)
            }
        }
    }

    private func failSession(_ connection: NWConnection, operations: AdapterOperations, message: String) {
        let wasCurrent = stateLock.withLock {
            guard self.connection === connection else { return false }
            self.connection = nil
            return true
        }
        connection.cancel()
        operations.close()
        if wasCurrent { onDisconnect?(message) }
    }

    private func sendAdapterPayload(_ payload: Sage_Ipc_V2_Frame.OneOf_Payload, connection: NWConnection) async throws {
        var frame = Sage_Ipc_V2_Frame()
        frame.protocolVersion = 2
        frame.payload = payload
        try await writeFrame(frame, connection: connection)
    }

    private func sendAdapterFailure(_ id: String, _ message: String, connection: NWConnection) async throws {
        var response = Sage_Ipc_V2_AdapterResult()
        response.requestID = id
        response.error = message
        try await sendAdapterPayload(.adapterResult(response), connection: connection)
    }

    private func writeFrame(_ frame: Sage_Ipc_V2_Frame, connection: NWConnection) async throws {
        try await writer.write(encode: { sequence in
            guard self.stateLock.withLock({ self.connection === connection }) else {
                throw SageClientError.connectionFailed("The IPC session changed before this request was sent")
            }
            var outbound = frame
            outbound.sequence = sequence
            let payload = try outbound.serializedData()
            guard !payload.isEmpty, payload.count <= 4 * 1024 * 1024 else {
                throw SageClientError.protocolError("IPC frame is outside the accepted size range")
            }
            var length = UInt32(payload.count).bigEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }
            data.append(payload)
            return data
        }, transmit: { data, complete in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error {
                    complete(.failure(SageClientError.connectionFailed(error.localizedDescription)))
                } else {
                    complete(.success(()))
                }
            })
        })
    }

    private func receiveFrame(_ connection: NWConnection) async throws -> Sage_Ipc_V2_Frame {
        let header = try await receiveExactly(4, connection: connection)
        let length = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).bigEndian }
        guard length > 0, length <= 4 * 1024 * 1024 else {
            throw SageClientError.protocolError("SAGE Core sent an invalid frame length")
        }
        let payload = try await receiveExactly(Int(length), connection: connection)
        return try Sage_Ipc_V2_Frame(serializedBytes: payload)
    }

    private func receiveExactly(_ count: Int, connection: NWConnection) async throws -> Data {
        var result = Data()
        while result.count < count {
            let remaining = count - result.count
            let chunk = try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Data, Error>) in
                connection.receive(
                    minimumIncompleteLength: 1,
                    maximumLength: remaining
                ) { content, _, complete, error in
                    if let error {
                        continuation.resume(throwing: SageClientError.connectionFailed(error.localizedDescription))
                    } else if let content, !content.isEmpty {
                        continuation.resume(returning: content)
                    } else if complete {
                        continuation.resume(throwing: SageClientError.connectionFailed("SAGE Core closed the connection"))
                    } else {
                        continuation.resume(throwing: SageClientError.protocolError("IPC receive returned no bytes"))
                    }
                }
            }
            result.append(chunk)
        }
        return result
    }

    private static func socketPath() throws -> String {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return support.appendingPathComponent("Sage/sage-core.sock").path
    }

    private static func proof(
        secret: Data,
        serverNonce: Data,
        clientNonce: Data,
        protocolVersion: UInt32,
        clientKind: Int,
        clientVersion: String,
        features: [String]
    ) -> Data {
        var message = Data("SAGE-LOCAL-IPC-AUTH-V2\0".utf8)
        message.append(serverNonce)
        message.append(clientNonce)
        var protocolValue = protocolVersion.bigEndian
        withUnsafeBytes(of: &protocolValue) { message.append(contentsOf: $0) }
        var kindValue = Int32(clientKind).bigEndian
        withUnsafeBytes(of: &kindValue) { message.append(contentsOf: $0) }
        let versionBytes = Data(clientVersion.utf8)
        var versionLength = UInt32(versionBytes.count).bigEndian
        withUnsafeBytes(of: &versionLength) { message.append(contentsOf: $0) }
        message.append(versionBytes)
        appendFeatureTranscript(features, to: &message)
        let key = SymmetricKey(data: secret)
        return Data(HMAC<SHA256>.authenticationCode(for: message, using: key))
    }

    private static func appendFeatureTranscript(_ features: [String], to message: inout Data) {
        guard !features.isEmpty else { return }
        message.append(Data("SAGE-IPC-FEATURES-V1\0".utf8))
        var count = UInt32(features.count).bigEndian
        withUnsafeBytes(of: &count) { message.append(contentsOf: $0) }
        for feature in features {
            let bytes = Data(feature.utf8)
            var length = UInt32(bytes.count).bigEndian
            withUnsafeBytes(of: &length) { message.append(contentsOf: $0) }
            message.append(bytes)
        }
    }
}

private final class ContinuationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    @discardableResult
    func resume() -> Bool {
        let pending = lock.withLock {
            let pending = continuation
            continuation = nil
            return pending
        }
        pending?.resume()
        return pending != nil
    }

    @discardableResult
    func resume(throwing error: Error) -> Bool {
        let pending = lock.withLock {
            let pending = continuation
            continuation = nil
            return pending
        }
        pending?.resume(throwing: error)
        return pending != nil
    }
}
