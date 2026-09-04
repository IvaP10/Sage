import CryptoKit
import Foundation
import Network
import SwiftProtobuf

enum SageClientError: LocalizedError {
    case connectionFailed(String)
    case authenticationFailed(String)
    case protocolError(String)

    var errorDescription: String? {
        switch self {
        case .connectionFailed(let message), .authenticationFailed(let message), .protocolError(let message):
            message
        }
    }
}

final class SageCoreClient: @unchecked Sendable {
    var onEvent: (@Sendable (Sage_Ipc_V1_CoreEvent) -> Void)?

    private let queue = DispatchQueue(label: "com.ivanpadeliya.sage.ipc")
    private let stateLock = NSLock()
    private var connection: NWConnection?
    private var sequence: UInt64 = 0

    func connect() async throws {
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
        stateLock.withLock { self.connection = connection }
        try await authenticate(connection)
        receiveLoop(connection)
    }

    func disconnect() {
        stateLock.withLock {
            connection?.cancel()
            connection = nil
        }
    }

    func submitTask(
        _ text: String,
        source: Sage_Ipc_V1_InputSource = .typed
    ) async throws {
        var submit = Sage_Ipc_V1_SubmitTask()
        submit.text = text
        submit.source = source
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .submitTask(submit)
        try await send(command)
    }

    func requestState(includeCompleted: Bool) async throws {
        var request = Sage_Ipc_V1_GetState()
        request.includeCompletedTasks = includeCompleted
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .getState(request)
        try await send(command)
    }

    func resolveApproval(
        _ approval: Sage_Ipc_V1_ApprovalRequest,
        approve: Bool,
        nativeAuthenticationSatisfied: Bool
    ) async throws {
        var response = Sage_Ipc_V1_ApprovalResponse()
        response.taskID = approval.taskID
        response.actionID = approval.actionID
        response.approvalID = approval.approvalID
        response.approvalDigest = approval.approvalDigest
        response.decision = approve ? .approveOnce : .deny
        response.nativeAuthenticationSatisfied = nativeAuthenticationSatisfied
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .approvalResponse(response)
        try await send(command)
    }

    func answer(_ question: Sage_Ipc_V1_QuestionRequest, text: String) async throws {
        var answer = Sage_Ipc_V1_UserAnswer()
        answer.taskID = question.taskID
        answer.actionID = question.actionID
        answer.questionID = question.questionID
        answer.answer = text
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .userAnswer(answer)
        try await send(command)
    }

    func control(taskID: String, operation: Sage_Ipc_V1_ControlTask.Operation) async throws {
        var control = Sage_Ipc_V1_ControlTask()
        control.taskID = taskID
        control.operation = operation
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .controlTask(control)
        try await send(command)
    }

    func undo(taskID: String) async throws {
        var undo = Sage_Ipc_V1_UndoLastAction()
        undo.taskID = taskID
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .undoLastAction(undo)
        try await send(command)
    }

    func saveProviderSettings(
        provider: String,
        model: String,
        endpoint: String,
        apiKey: String,
        removeSavedKey: Bool,
        nativeAuthenticationSatisfied: Bool
    ) async throws {
        var settings = Sage_Ipc_V1_SaveProviderSettings()
        settings.role = "reasoning"
        settings.provider = provider
        settings.model = model
        settings.endpoint = endpoint
        settings.apiKey = apiKey
        settings.removeSavedKey = removeSavedKey
        settings.nativeAuthenticationSatisfied = nativeAuthenticationSatisfied
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .saveProviderSettings(settings)
        try await send(command)
    }

    func testProviderConnection(
        provider: String,
        model: String,
        endpoint: String,
        apiKey: String
    ) async throws {
        var settings = Sage_Ipc_V1_TestProviderConnection()
        settings.role = "reasoning"
        settings.provider = provider
        settings.model = model
        settings.endpoint = endpoint
        settings.apiKey = apiKey
        var command = Sage_Ipc_V1_UiCommand()
        command.requestID = UUID().uuidString
        command.command = .testProviderConnection(settings)
        try await send(command)
    }

    private func authenticate(_ connection: NWConnection) async throws {
        let challengeFrame = try await receiveFrame(connection)
        guard challengeFrame.protocolVersion == 1,
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
        var authentication = Sage_Ipc_V1_ClientAuthenticate()
        authentication.clientKind = .macos
        authentication.clientVersion = version
        authentication.clientNonce = clientNonce
        authentication.proof = Self.proof(
            secret: secret,
            serverNonce: challenge.nonce,
            clientNonce: clientNonce,
            protocolVersion: 1,
            clientKind: Sage_Ipc_V1_ClientKind.macos.rawValue,
            clientVersion: version
        )
        var frame = Sage_Ipc_V1_Frame()
        frame.protocolVersion = 1
        frame.sequence = nextSequence()
        frame.payload = .clientAuthenticate(authentication)
        try await writeFrame(frame, connection: connection)
        let resultFrame = try await receiveFrame(connection)
        guard case .authenticationResult(let result) = resultFrame.payload, result.accepted else {
            throw SageClientError.authenticationFailed("SAGE Core rejected local IPC authentication")
        }
    }

    private func send(_ command: Sage_Ipc_V1_UiCommand) async throws {
        guard let connection = stateLock.withLock({ self.connection }) else {
            throw SageClientError.connectionFailed("SAGE Core is not connected")
        }
        var frame = Sage_Ipc_V1_Frame()
        frame.protocolVersion = 1
        frame.sequence = nextSequence()
        frame.payload = .uiCommand(command)
        try await writeFrame(frame, connection: connection)
    }

    private func receiveLoop(_ connection: NWConnection) {
        Task.detached { [weak self] in
            do {
                while !Task.isCancelled {
                    let frame = try await self?.receiveFrame(connection)
                    guard let frame else { return }
                    if case .coreEvent(let event) = frame.payload {
                        self?.onEvent?(event)
                    }
                }
            } catch {
                self?.disconnect()
            }
        }
    }

    private func writeFrame(_ frame: Sage_Ipc_V1_Frame, connection: NWConnection) async throws {
        let payload = try frame.serializedData()
        guard !payload.isEmpty, payload.count <= 4 * 1024 * 1024 else {
            throw SageClientError.protocolError("IPC frame is outside the accepted size range")
        }
        var length = UInt32(payload.count).bigEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(payload)
        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: SageClientError.connectionFailed(error.localizedDescription))
                } else {
                    continuation.resume()
                }
            })
        }
    }

    private func receiveFrame(_ connection: NWConnection) async throws -> Sage_Ipc_V1_Frame {
        let header = try await receiveExactly(4, connection: connection)
        let length = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).bigEndian }
        guard length > 0, length <= 4 * 1024 * 1024 else {
            throw SageClientError.protocolError("SAGE Core sent an invalid frame length")
        }
        let payload = try await receiveExactly(Int(length), connection: connection)
        return try Sage_Ipc_V1_Frame(serializedBytes: payload)
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

    private func nextSequence() -> UInt64 {
        stateLock.withLock {
            sequence &+= 1
            return sequence
        }
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
        clientVersion: String
    ) -> Data {
        var message = Data("SAGE-LOCAL-IPC-AUTH-V1\0".utf8)
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
        let key = SymmetricKey(data: secret)
        return Data(HMAC<SHA256>.authenticationCode(for: message, using: key))
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
