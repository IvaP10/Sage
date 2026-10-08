import SwiftUI

struct ControllerReviewView: View {
    let draft: ControllerDraftRecord
    let busy: Bool
    let status: String
    let review: () -> Void
    let later: () -> Void

    private var steps: [ControllerDraftStep] { draft.steps ?? [] }
    private var hasCompletePreview: Bool { steps.count == draft.stepCount && !steps.isEmpty }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Review controller draft")
                    .font(.system(size: 20, weight: .semibold))
                Text("\(draft.systemLabel) · \(draft.stepCount) steps · revision \(draft.revision)")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }

            Text("Review the exact saved procedure below. Sage will passively inspect the current app and check every semantic control. It will not activate controls, run this procedure, or grant future permission.")
                .font(.system(size: 13))
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            if hasCompletePreview {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(Array(steps.enumerated()), id: \.element.id) { item in
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Step \(item.offset + 1): \(item.element.controlName)")
                                    .font(.system(size: 13, weight: .semibold))
                                    .textSelection(.enabled)
                                Text("Control: \(item.element.role)")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.secondary)
                                detail("Effect", item.element.expectedEffect)
                                detail("Verification", item.element.verification)
                                if let restoration = item.element.restoration, !restoration.isEmpty {
                                    detail("Restoration", restoration)
                                }
                            }
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                    .padding(.vertical, 2)
                }
            } else {
                Text("Sage could not load the complete bounded step preview. Review is unavailable until the draft can be inspected safely.")
                    .font(.system(size: 13))
                    .foregroundStyle(SageTheme.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if draft.status == "reviewed" {
                Label("Previously reviewed. Rechecking will use a new foreground observation.", systemImage: "checkmark.shield")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            if !status.isEmpty {
                Text(status)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Later", action: later)
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button(draft.status == "reviewed" ? "Revalidate current app" : "Review and rebind", action: review)
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy || !hasCompletePreview)
            }
        }
        .padding(24)
        .frame(minWidth: 520, idealWidth: 680, maxWidth: 760, minHeight: 420, idealHeight: 560, maxHeight: 720)
        .preferredColorScheme(.dark)
    }

    private func detail(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 12))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
