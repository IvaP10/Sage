import AppKit

@MainActor
final class GlobalShortcutController {
    private var monitor: Any?

    func install(action: @escaping @MainActor () -> Void) {
        uninstall()
        monitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { event in
            guard event.modifierFlags.intersection(.deviceIndependentFlagsMask) == [.command, .shift],
                  event.charactersIgnoringModifiers == " " else { return }
            Task { @MainActor in action() }
        }
    }

    func uninstall() {
        if let monitor {
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }
    }
}
