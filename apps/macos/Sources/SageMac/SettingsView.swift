import SwiftUI

struct SettingsView: View {
    @Bindable var model: AppModel
    @State private var provider = "openai"
    @State private var modelName = "gpt-5.4"
    @State private var endpoint = ""
    @State private var apiKey = ""
    @State private var removeSavedKey = false
    @State private var wakePhraseDraft = "Hey Sage"
    @State private var savedProviderDraft: ProviderDraft?
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case provider
        case model
        case endpoint
        case apiKey
        case wakePhrase
    }

    private struct ProviderDraft: Equatable {
        let provider: String
        let model: String
        let endpoint: String
        let hasNewAPIKey: Bool
        let removesSavedKey: Bool
    }

    private let providers = [
        ("openai", "OpenAI"),
        ("openai-compatible", "OpenAI-compatible endpoint"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            settingsHeader
            Rectangle()
                .fill(SageTheme.stroke)
                .frame(height: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    connectionCard
                    modelCard
                    voiceCard
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
        .onChange(of: model.providerSettings?.provider) {
            loadProviderSettings()
        }
        .onChange(of: model.providerSaveMessage) {
            if model.providerSaveMessage != nil {
                savedProviderDraft = currentProviderDraft
                removeSavedKey = false
            }
        }
        .onChange(of: currentProviderDraft) {
            if !model.providerSaving {
                model.providerSaveMessage = nil
            }
        }
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

    private var modelCard: some View {
        SettingsCard(
            title: "Model",
            systemImage: "sparkles"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top, spacing: 14) {
                    settingsField("Provider", field: .provider) {
                        providerMenu
                    }
                    settingsField("Model", field: .model) {
                        TextField("Model name", text: $modelName)
                            .textFieldStyle(.plain)
                            .focused($focusedField, equals: .model)
                    }
                }

                settingsField("Endpoint", field: .endpoint) {
                    TextField(
                        provider == "openai-compatible" ? "https://your-host/v1" : "https://api.openai.com/v1 (optional)",
                        text: $endpoint
                    )
                    .textFieldStyle(.plain)
                    .focused($focusedField, equals: .endpoint)
                }

                settingsField("API key", field: .apiKey) {
                    SecureField(
                        "API key",
                        text: $apiKey
                    )
                    .textFieldStyle(.plain)
                    .focused($focusedField, equals: .apiKey)
                }

                if model.providerSettings?.hasApiKey_p == true {
                    Toggle("Remove the saved Keychain credential", isOn: $removeSavedKey)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 12))
                }

                HStack(spacing: 10) {
                    if let message = model.providerSaveMessage {
                        Label(message, systemImage: "checkmark.circle.fill")
                            .font(.system(size: 11.5, weight: .medium))
                            .foregroundStyle(SageTheme.success)
                    } else if model.providerSettings?.hasApiKey_p == true {
                        Label("Credential saved in Keychain", systemImage: "key.fill")
                            .font(.system(size: 11.5))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(model.providerTesting ? "Testing…" : "Test connection") {
                        model.testProviderConnection(
                            provider: provider,
                            model: modelName,
                            endpoint: endpoint,
                            apiKey: apiKey
                        )
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.providerTesting || !validProviderDraft)
                    .help("Test the provider without saving the credential")
                    Button(model.providerSaving ? "Saving…" : "Save model settings") {
                        let key = apiKey
                        apiKey = ""
                        model.saveProviderSettings(
                            provider: provider,
                            model: modelName,
                            endpoint: endpoint,
                            apiKey: key,
                            removeSavedKey: removeSavedKey
                        )
                        removeSavedKey = false
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        model.providerSaving
                            || !hasModelChanges
                            || !validProviderDraft
                    )
                    .help("Save changed model settings")
                }
                if let message = model.providerTestMessage {
                    Text(message)
                        .font(.system(size: 11.5))
                        .foregroundStyle(message.hasPrefix("Connected") ? SageTheme.success : SageTheme.danger)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var voiceCard: some View {
        SettingsCard(
            title: "Voice",
            systemImage: "waveform"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Listen for a wake word")
                        .font(.system(size: 13, weight: .medium))
                    Spacer()
                    Toggle("", isOn: Binding(
                        get: { model.wakeWordEnabled },
                        set: { enabled in model.setWakeWordEnabled(enabled) }
                    ))
                    .labelsHidden()
                    .toggleStyle(.switch)
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

    private var providerMenu: some View {
        Menu {
            ForEach(providers, id: \.0) { value, label in
                Button {
                    provider = value
                } label: {
                    if provider == value {
                        Label(label, systemImage: "checkmark")
                    } else {
                        Text(label)
                    }
                }
            }
        } label: {
            Text(selectedProviderLabel)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .buttonStyle(.plain)
        .focused($focusedField, equals: .provider)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var selectedProviderLabel: String {
        providers.first(where: { $0.0 == provider })?.1
            ?? (provider.isEmpty ? "Unconfigured provider" : "Unconfigured legacy provider (\(provider))")
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
        loadProviderSettings()
        savedProviderDraft = currentProviderDraft
    }

    private func loadProviderSettings() {
        guard let saved = model.providerSettings else { return }
        provider = saved.provider
        modelName = saved.model
        endpoint = saved.endpoint
        savedProviderDraft = currentProviderDraft
    }

    private var currentProviderDraft: ProviderDraft {
        ProviderDraft(
            provider: provider,
            model: modelName.trimmingCharacters(in: .whitespacesAndNewlines),
            endpoint: endpoint.trimmingCharacters(in: .whitespacesAndNewlines),
            hasNewAPIKey: !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            removesSavedKey: removeSavedKey
        )
    }

    private var hasModelChanges: Bool {
        guard let savedProviderDraft else { return false }
        return currentProviderDraft != savedProviderDraft
    }

    private var validProviderDraft: Bool {
        !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (provider != "openai-compatible"
                || !endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
