import Foundation

/// Retains identities until the core confirms durable acceptance. This bounded
/// in-memory outbox survives a transport reconnect, not an application restart.
final class PendingSubmissions: @unchecked Sendable {
    enum Failure: LocalizedError {
        case full
        var errorDescription: String? { "Waiting for the core to confirm earlier requests. Reconnect before sending more." }
    }
    struct Entry: Sendable {
        let requestID: String
        let payload: Data
    }
    private let lock = NSLock()
    private var entries: [Entry] = []
    private let maximumEntries = 16
    private let maximumBytes = 1024 * 1024

    func stage(_ payload: Data, requestID: String? = nil) throws -> Entry {
        try lock.withLock {
            if let existing = entries.first(where: { $0.payload == payload }) { return existing }
            guard entries.count < maximumEntries,
                  payload.count <= maximumBytes - entries.reduce(0, { $0 + $1.payload.count }) else {
                throw Failure.full
            }
            let entry = Entry(requestID: requestID ?? UUID().uuidString.lowercased(), payload: payload)
            entries.append(entry)
            return entry
        }
    }

    func resolve(_ requestID: String) {
        lock.withLock { entries.removeAll { $0.requestID == requestID } }
    }

    func snapshot() -> [Entry] { lock.withLock { entries } }
}
