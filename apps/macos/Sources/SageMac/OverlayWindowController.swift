import AppKit
import SwiftUI

enum VoiceOverlayPhase {
    case listening
    case processing
    case executing
    case success
    case error

    var label: String {
        switch self {
        case .listening: "Listening…"
        case .processing: "Understanding…"
        case .executing: "Working…"
        case .success: "Done"
        case .error: "Something went wrong"
        }
    }

    var systemImage: String {
        switch self {
        case .listening: "waveform"
        case .processing: "hourglass"
        case .executing: "arrow.trianglehead.2.clockwise.rotate.90"
        case .success: "checkmark"
        case .error: "exclamationmark"
        }
    }

    var hideDelay: Duration? {
        switch self {
        case .success: .milliseconds(1_500)
        case .error: .milliseconds(2_500)
        case .listening, .processing, .executing: nil
        }
    }
}

@MainActor
final class OverlayWindowController {
    private var panel: NSPanel?
    private var status = OverlayStatus()
    private var hideTask: Task<Void, Never>?

    func show(phase: VoiceOverlayPhase, text: String? = nil) {
        hideTask?.cancel()
        status.phase = phase
        status.text = text?.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty ?? phase.label

        let panel = panel ?? makePanel()
        position(panel)
        panel.orderFrontRegardless()

        if let delay = phase.hideDelay {
            hideTask = Task { [weak self] in
                try? await Task.sleep(for: delay)
                guard !Task.isCancelled else { return }
                self?.hide()
            }
        }
    }

    func hide() {
        hideTask?.cancel()
        hideTask = nil
        panel?.orderOut(nil)
    }

    private func makePanel() -> NSPanel {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 440, height: 82),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = true
        panel.contentView = NSHostingView(rootView: OverlayView(status: status))
        self.panel = panel
        return panel
    }

    private func position(_ panel: NSPanel) {
        guard let visible = NSScreen.main?.visibleFrame else { return }
        panel.setFrameOrigin(
            NSPoint(
                x: visible.midX - panel.frame.width / 2,
                y: visible.minY + 34
            )
        )
    }
}

@MainActor
@Observable
private final class OverlayStatus {
    var phase: VoiceOverlayPhase = .listening
    var text = VoiceOverlayPhase.listening.label
}

private struct OverlayView: View {
    @Bindable var status: OverlayStatus

    var body: some View {
        HStack(spacing: 12) {
            SageLogo(size: 32)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Image(systemName: status.phase.systemImage)
                        .font(.system(size: 10, weight: .semibold))
                    Text(status.phase.label.replacingOccurrences(of: "…", with: ""))
                        .font(.system(size: 10, weight: .semibold))
                        .textCase(.uppercase)
                        .tracking(0.55)
                }
                .foregroundStyle(.secondary)
                Text(status.text)
                    .font(.system(size: 13.5, weight: .semibold))
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 15)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(.regularMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.15), lineWidth: 1)
                }
        }
        .padding(4)
    }
}

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}
