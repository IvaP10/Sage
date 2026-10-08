import SwiftUI

struct SettingsView: View {
    @Bindable var model: AppModel
    @State private var wakePhraseDraft = "Hey Sage"
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case wakePhrase
    }

    var body: some View {
        VStack(spacing: 0) {
            settingsHeader
            Rectangle()
                .fill(SageTheme.stroke)
                .frame(height: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    connectionCard
                    localInferenceCard
                    voiceCard
                    MemorySettingsView(model: model)
                    WorkflowSettingsView(model: model)
                    permissionsCard
                    localDataCard
                }
                .frame(maxWidth: 760)
                .padding(.horizontal, 34)
                .padding(.vertical, 28)
                .frame(maxWidth: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(SageTheme.canvas)
        .onAppear(perform: loadSettings)
    }

    private var settingsHeader: some View {
        HStack(spacing: 16) {
            Button {
                model.settingsVisible = false
            } label: {
                Label("Back", systemImage: "chevron.left")
            }
            .buttonStyle(SageBorderedButtonStyle())
            .keyboardShortcut(.cancelAction)
            .help("Back to chats")

            Text("Settings")
                .font(.system(size: 22, weight: .semibold))
            Spacer()
        }
        .padding(.horizontal, 30)
        .frame(height: 72)
    }

    private var connectionCard: some View {
        SettingsCard(
            title: "Local core",
            systemImage: "cpu"
        ) {
            StatusRow(
                title: "Connection",
                value: model.connectionState.label,
                positive: model.connectionState == .connected
            )
            Divider()
            StatusRow(title: "Storage", value: "Local SQLite", positive: true)
        }
    }

    private var localInferenceCard: some View {
        SettingsCard(
            title: "Sage local inference",
            systemImage: "sparkles"
        ) {
            StatusRow(title: "Target", value: "Qwen3.5-4B", positive: nil)
            Divider()
            StatusRow(title: "Engine", value: "In development", positive: false)
            Text("Sage is building its own tokenizer, weight reader, CPU and Metal kernels, and persistent inference worker. Model-based tasks stay unavailable until that path passes numerical, quality, memory, latency, and hardware qualification.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var voiceCard: some View {
        SettingsCard(
            title: "Voice",
            systemImage: "waveform"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Spacer()
                    Toggle("Enable wake word", isOn: Binding(
                        get: { model.wakeWordEnabled },
                        set: { enabled in model.setWakeWordEnabled(enabled) }
                    ))
                    .labelsHidden()
                    .toggleStyle(.switch)
                }

                if model.isSpeaking {
                    HStack(spacing: 8) {
                        Label("Speaking", systemImage: "speaker.wave.2.fill")
                        Spacer()
                        Button("Stop speaking", action: model.stopSpokenReply)
                            .buttonStyle(SageBorderlessButtonStyle())
                            .accessibilityLabel("Stop spoken reply")
                    }
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(.secondary)
                }

                settingsField("Wake phrase", field: .wakePhrase) {
                    HStack {
                        TextField("Hey Sage", text: $wakePhraseDraft)
                            .textFieldStyle(.plain)
                            .focused($focusedField, equals: .wakePhrase)
                            .onSubmit { model.setWakePhrase(wakePhraseDraft) }
                        Button("Update") {
                            model.setWakePhrase(wakePhraseDraft)
                            wakePhraseDraft = model.wakePhrase
                        }
                        .buttonStyle(SageBorderlessButtonStyle())
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(SageTheme.accent)
                        .disabled(
                            wakePhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                || wakePhraseDraft.trimmingCharacters(in: .whitespacesAndNewlines) == model.wakePhrase
                        )
                        .help("Update wake phrase")
                    }
                }
            }
        }
    }

    private var permissionsCard: some View {
        SettingsCard(
            title: "Permissions",
            systemImage: "hand.raised"
        ) {
            StatusRow(
                title: "Microphone",
                value: model.microphonePermissionLabel,
                positive: model.microphonePermissionLabel == "Allowed"
            )
            Divider()
            StatusRow(
                title: "Speech recognition",
                value: model.speechPermissionLabel,
                positive: model.speechPermissionLabel == "Allowed"
            )
            Divider()
            StatusRow(
                title: "Keychain",
                value: "Protected",
                positive: nil
            )
            Divider()
            StatusRow(
                title: "Accessibility",
                value: "On demand",
                positive: nil
            )
        }
    }

    private var localDataCard: some View {
        SettingsCard(
            title: "Local data",
            systemImage: "internaldrive"
        ) {
            StatusRow(title: "Storage", value: "Application Support", positive: nil)
            Divider()
            StatusRow(title: "IPC key", value: "Owner-only", positive: nil)
        }
    }

    private func settingsField<Content: View>(
        _ label: String,
        field: Field,
        @ViewBuilder content: () -> Content
    ) -> some View {
        SageSettingsField(label: label, isFocused: focusedField == field) {
            content()
        }
    }

    private func loadSettings() {
        wakePhraseDraft = model.wakePhrase
    }
}

private struct SageSettingsField<Content: View>: View {
    let label: String
    let isFocused: Bool
    @ViewBuilder let content: Content
    @State private var isHovering = false

    init(label: String, isFocused: Bool, @ViewBuilder content: () -> Content) {
        self.label = label
        self.isFocused = isFocused
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(label)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.secondary)
            content
                .padding(.horizontal, 11)
                .frame(minHeight: 38)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    isFocused || isHovering ? SageTheme.inputFocusedFill : SageTheme.inputFill,
                    in: RoundedRectangle(cornerRadius: 9, style: .continuous)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .stroke(isFocused ? SageTheme.accent.opacity(0.36) : (isHovering ? SageTheme.strongStroke : SageTheme.stroke), lineWidth: 1)
                }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onHover { isHovering = $0 }
        .animation(.easeOut(duration: 0.14), value: isHovering)
    }
}

private struct SageBorderlessButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 7)
            .padding(.vertical, 5)
            .background(
                configuration.isPressed ? SageTheme.selectionFill : Color.clear,
                in: RoundedRectangle(cornerRadius: 7, style: .continuous)
            )
            .opacity(configuration.isPressed ? 0.78 : 1)
    }
}

private struct SettingsCard<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    init(
        title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 11) {
                Image(systemName: systemImage)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 30, height: 30)
                    .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 9))
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
            }
            content
        }
        .padding(18)
        .background(SageTheme.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(SageTheme.stroke, lineWidth: 1)
        }
    }
}

private struct StatusRow: View {
    let title: String
    let value: String
    let positive: Bool?

    var body: some View {
        HStack(spacing: 12) {
            Text(title)
                .font(.system(size: 12.5, weight: .medium))
            Spacer()
            if let positive {
                Circle()
                    .fill(positive ? SageTheme.success : SageTheme.warning)
                    .frame(width: 6, height: 6)
            }
            Text(value)
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
        .padding(.vertical, 2)
    }
}
