import Foundation

@main
struct PresentationRefreshSmoke {
    @MainActor static func main() async throws {
        var sends = 0
        var failures = 0
        let refresh = PresentationRefreshCoordinator(delay: .milliseconds(10), timeout: .milliseconds(120), send: {
            sends += 1
        }, failed: { _ in failures += 1 })
        for _ in 0..<1000 { refresh.request() }
        try await Task.sleep(for: .milliseconds(40))
        precondition(sends == 1, "A burst should produce one snapshot request")
        for _ in 0..<1000 { refresh.request() }
        try await Task.sleep(for: .milliseconds(20))
        precondition(sends == 1, "Only one snapshot may be in flight")
        refresh.received()
        try await Task.sleep(for: .milliseconds(40))
        precondition(sends == 2, "In-flight invalidations need one trailing refresh")
        refresh.received()
        try await Task.sleep(for: .milliseconds(40))
        precondition(sends == 2 && failures == 0, "A clean response must not loop")
        refresh.request()
        refresh.reset()
        try await Task.sleep(for: .milliseconds(30))
        precondition(sends == 2, "Disconnect must cancel scheduled refresh")
        refresh.request(immediate: true)
        try await Task.sleep(for: .milliseconds(160))
        precondition(sends == 3 && failures == 1, "A missing response must time out exactly once")
        refresh.request(immediate: true)
        try await Task.sleep(for: .milliseconds(10))
        refresh.received()
        precondition(sends == 4, "A timed-out lane must be reusable")
        print("Presentation refresh: bursts, in-flight invalidation, disconnect, timeout, recovery passed")
    }
}
