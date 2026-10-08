import Foundation

/// One bounded refresh lane. Deltas remain immediate; a burst of invalidations
/// produces one request and, if state changed while it was in flight, one follow-up.
@MainActor
final class PresentationRefreshCoordinator {
    private let delay: Duration
    private let timeout: Duration
    private let send: @MainActor () async throws -> Void
    private let failed: @MainActor (Error) -> Void
    private var scheduled: Task<Void, Never>?
    private var deadline: Task<Void, Never>?
    private var dirty = false
    private var inFlight = false
    private var generation: UInt = 0

    init(
        delay: Duration = .milliseconds(180),
        timeout: Duration = .seconds(8),
        send: @escaping @MainActor () async throws -> Void,
        failed: @escaping @MainActor (Error) -> Void
    ) {
        self.delay = delay
        self.timeout = timeout
        self.send = send
        self.failed = failed
    }

    func request(immediate: Bool = false) {
        dirty = true
        guard !inFlight else { return }
        if immediate { scheduled?.cancel(); scheduled = nil }
        guard scheduled == nil else { return }
        let epoch = generation
        scheduled = Task { [weak self] in
            guard let self else { return }
            if !immediate {
                do { try await Task.sleep(for: delay) } catch { return }
            }
            guard !Task.isCancelled, generation == epoch else { return }
            scheduled = nil
            dirty = false
            inFlight = true
            deadline = Task { [weak self] in
                guard let self else { return }
                do { try await Task.sleep(for: timeout) } catch { return }
                guard generation == epoch, inFlight else { return }
                reset()
                failed(RefreshError.timedOut)
            }
            do {
                try await send()
            } catch {
                guard generation == epoch else { return }
                reset()
                failed(error)
            }
        }
    }

    func received() {
        guard inFlight else { return }
        inFlight = false
        deadline?.cancel()
        deadline = nil
        if dirty { request() }
    }

    func reset() {
        generation &+= 1
        scheduled?.cancel()
        scheduled = nil
        deadline?.cancel()
        deadline = nil
        dirty = false
        inFlight = false
    }

    private enum RefreshError: LocalizedError {
        case timedOut
        var errorDescription: String? {
            "Sage could not refresh its state. Reconnect to check the latest task result."
        }
    }
}
