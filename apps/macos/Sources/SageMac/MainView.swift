import SwiftUI

struct MainView: View {
    @Bindable var model: AppModel
    @FocusState private var composerFocused: Bool
    @FocusState private var renameFocused: Bool
    @State private var questionAnswer = ""
    @State private var composerHovering = false
    @State private var hoveredTaskID: String?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Rectangle()
                .fill(SageTheme.stroke)
                .frame(width: 1)
            if model.settingsVisible {
                SettingsView(model: model)
            } else {
                workspace
            }
        }
        .frame(minWidth: 920, minHeight: 640)
        .background(SageTheme.canvas)
        .onChange(of: model.composerFocusToken) {
            composerFocused = true
        }
        .onChange(of: model.renameFocusToken) {
            renameFocused = true
        }
        .onChange(of: renameFocused) {
            if !renameFocused {
                model.commitRename()
            }
        }
        .alert("Error", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
        .alert("Delete conversation?", isPresented: Binding(
            get: { model.deleteCandidateID != nil },
            set: { if !$0 { model.cancelDelete() } }
        )) {
            Button("Cancel", role: .cancel) { model.cancelDelete() }
            Button("Delete", role: .destructive) { model.confirmDelete() }
        } message: {
            Text("Remove “\(model.deleteCandidateTitle)” from Recent chats?")
        }
        .sheet(item: $model.pendingApproval) { approval in
            ApprovalView(
                approval: approval,
                approve: { model.approve(approval) },
                deny: { model.deny(approval) }
            )
            .interactiveDismissDisabled()
        }
        .sheet(item: $model.pendingQuestion) { question in
            questionSheet(question)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            sidebarHeader

            Button(action: model.newTask) {
                HStack(spacing: 10) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 13, weight: .medium))
                    Text("New chat")
                        .font(.system(size: 13, weight: .medium))
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 12)
                .frame(height: 36)
                .contentShape(Rectangle())
            }
            .buttonStyle(SageSidebarButtonStyle())
            .focusEffectDisabled()
            .padding(.horizontal, 10)
            .padding(.top, 10)

            Text("Recent")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 18)
                .padding(.top, 22)
                .padding(.bottom, 8)

            ScrollView(.vertical) {
                LazyVStack(spacing: 2) {
                    if model.visibleTasks.isEmpty {
                        Text("No chats yet")
                            .font(.system(size: 12))
                            .foregroundStyle(.tertiary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(model.visibleTasks, id: \.taskID) { task in
                            taskRow(task)
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .frame(maxHeight: .infinity)
            .scrollIndicators(.automatic)

            if model.voiceState == .wakeListening {
                HStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .symbolEffect(.variableColor.iterative)
                    Text("Listening for “\(model.wakePhrase)”")
                        .lineLimit(1)
                }
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 18)
                .padding(.bottom, 10)
            }

            Button {
                model.settingsVisible = true
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "gearshape")
                        .font(.system(size: 13, weight: .medium))
                    Text("Settings")
                        .font(.system(size: 13, weight: .medium))
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 12)
                .frame(height: 36)
                .contentShape(Rectangle())
            }
            .buttonStyle(SageSidebarButtonStyle(selected: model.settingsVisible))
            .padding(.horizontal, 10)
            .padding(.bottom, 14)
        }
        .frame(minWidth: 210, idealWidth: 238, maxWidth: 280)
        .background(SageTheme.sidebar)
    }

    private var sidebarHeader: some View {
        HStack(spacing: 8) {
            SageLogo(size: 22)
            Text("Sage")
                .font(.system(size: 16, weight: .semibold))
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.top, 30)
        .padding(.bottom, 14)
    }

    private func taskRow(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        let selected = model.selectedTaskID == task.taskID && !model.settingsVisible
        let showingOptions = hoveredTaskID == task.taskID

        return Group {
            if model.editingTaskID == task.taskID {
                HStack(spacing: 6) {
                    TextField("Chat title", text: $model.editingTaskTitle)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12.5, weight: .medium))
                        .focused($renameFocused)
                        .onSubmit { model.commitRename() }
                        .onExitCommand { model.cancelRename() }
                    Button {
                        model.commitRename()
                    } label: {
                        Image(systemName: "checkmark")
                            .font(.system(size: 11, weight: .bold))
                            .frame(width: 24, height: 24)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(SageTheme.accent)
                    .help("Save title")
                }
                .padding(.horizontal, 10)
                .frame(minHeight: 42)
                .background(SageTheme.selectionFill, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            } else {
                ZStack(alignment: .trailing) {
                    Button {
                        model.selectTask(task.taskID)
                    } label: {
                        HStack(spacing: 7) {
                            if model.isPinned(task.taskID) {
                                Image(systemName: "pin.fill")
                                    .font(.system(size: 9.5, weight: .semibold))
                                    .foregroundStyle(.secondary)
                            }
                            Text(model.displayTitle(for: task))
                                .font(.system(size: 12.5, weight: .medium))
                                .lineLimit(1)
                                .truncationMode(.tail)
                            Spacer(minLength: 0)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.leading, 10)
                        .padding(.trailing, 38)
                        .padding(.vertical, 9)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(SageSidebarButtonStyle(
                        selected: selected,
                        externallyHovered: showingOptions
                    ))
                    .focusEffectDisabled()
                    .help(model.displayTitle(for: task))

                    taskOptionsMenu(task)
                        .padding(.trailing, 5)
                        .opacity(showingOptions ? 1 : 0)
                        .allowsHitTesting(showingOptions)
                        .accessibilityHidden(!showingOptions)
                }
                .onHover { hovering in
                    if hovering {
                        hoveredTaskID = task.taskID
                    } else if hoveredTaskID == task.taskID {
                        hoveredTaskID = nil
                    }
                }
                .animation(.easeOut(duration: 0.12), value: showingOptions)
            }
        }
        .frame(maxWidth: .infinity)
        .contextMenu {
            taskMenuItems(task)
        }
    }

    private func taskOptionsMenu(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        Menu {
            taskMenuItems(task)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
                .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .focusEffectDisabled()
        .help("Chat options")
        .accessibilityLabel("Options for \(model.displayTitle(for: task))")
    }

    @ViewBuilder
    private func taskMenuItems(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        Button("Rename", systemImage: "pencil") {
            model.beginRename(task)
        }
        Button(
            model.isPinned(task.taskID) ? "Unpin" : "Pin",
            systemImage: model.isPinned(task.taskID) ? "pin.slash" : "pin"
        ) {
            model.togglePinned(task.taskID)
        }
        Divider()
        Button("Delete", systemImage: "trash", role: .destructive) {
            model.requestDelete(task)
        }
    }

    private var workspace: some View {
        VStack(spacing: 0) {
            if let task = selectedTask {
                taskToolbar(task)
            }
            conversation
            composerArea
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(SageTheme.canvas)
    }

    private func taskToolbar(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayTitle(for: task))
                    .font(.system(size: 15.5, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 16)
            if task.undoAvailable {
                Button {
                    model.undo(taskID: task.taskID)
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                }
                .buttonStyle(SageBorderedButtonStyle())
                .help("Undo the last reversible action")
            }
            if !isFinished(task.status) {
                Button {
                    model.cancel(taskID: task.taskID)
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(SageBorderedButtonStyle(destructive: true))
                .help("Stop task")
            }
        }
        .padding(.horizontal, 32)
        .padding(.top, 17)
        .padding(.bottom, 11)
        .frame(maxWidth: .infinity)
    }

    private var conversation: some View {
        Group {
            if let task = selectedTask {
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        requestMessage(task)
                        if model.timeline.isEmpty {
                            taskResult(task)
                        } else {
                            ForEach(Array(model.timeline.reversed().enumerated()), id: \.offset) { _, event in
                                timelineEvent(event)
                            }
                        }
                    }
                    .frame(maxWidth: 920, alignment: .leading)
                    .padding(.horizontal, 32)
                    .padding(.top, 12)
                    .padding(.bottom, 18)
                    .frame(maxWidth: .infinity)
                }
                .scrollIndicators(.automatic)
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func requestMessage(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        HStack(alignment: .bottom) {
            Spacer(minLength: 48)
            Text(task.request)
                .font(.system(size: 14, weight: .medium))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 15)
                .padding(.vertical, 11)
                .background(SageTheme.userBubble, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        }
        .padding(.bottom, 18)
    }

    private func taskResult(_ task: Sage_Ipc_V1_TaskUpdate) -> some View {
        let detail = task.summary.isEmpty ? task.finalOutcome : task.summary

        return HStack(alignment: .top, spacing: 12) {
            SageLogo(size: 22)
                .shadow(color: Color.black.opacity(0.12), radius: 4, y: 2)
            VStack(alignment: .leading, spacing: 7) {
                Label(statusLabel(task.status), systemImage: statusSymbol(task.status))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(statusColor(task.status))
                if !detail.isEmpty {
                    Text(detail)
                        .font(.system(size: 13.5))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                } else if !task.currentAction.isEmpty {
                    Text(task.currentAction)
                        .font(.system(size: 13.5))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 12)
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            SageLogo(size: 46)
                .shadow(color: Color.black.opacity(0.16), radius: 12, y: 6)
            Text("What should we build?")
                .font(.system(size: 24, weight: .semibold))
        }
        .padding(.horizontal, 32)
        .padding(.bottom, 64)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func timelineEvent(_ event: Sage_Ipc_V1_AgentEvent) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: eventIcon(event.kind))
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 24, height: 24)
                .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            VStack(alignment: .leading, spacing: 5) {
                Text(event.title)
                    .font(.system(size: 13.5, weight: .semibold))
                if !event.detail.isEmpty {
                    Text(event.detail)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(SageTheme.stroke)
                .frame(height: 1)
                .padding(.leading, 36)
        }
    }

    private var composerArea: some View {
        VStack(spacing: 8) {
            if let notice = model.voiceNotice {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.circle")
                    Text(notice)
                        .lineLimit(2)
                    Spacer(minLength: 0)
                    Button("Dismiss") { model.voiceNotice = nil }
                        .buttonStyle(.plain)
                }
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .frame(maxWidth: 920)
            }

            VStack(alignment: .leading, spacing: 0) {
                if case .listening(let activation) = model.voiceState {
                    HStack(spacing: 9) {
                        Image(systemName: "waveform")
                            .foregroundStyle(SageTheme.accent)
                            .symbolEffect(.variableColor.iterative)
                        Text(model.voiceTranscript.isEmpty
                             ? (activation == .wakeWord ? "Wake word heard — listening…" : "Listening…")
                             : model.voiceTranscript)
                            .font(.system(size: 12.5, weight: .medium))
                            .lineLimit(2)
                        Spacer(minLength: 0)
                        Button("Cancel") { model.cancelVoiceInput() }
                            .font(.system(size: 11.5))
                            .buttonStyle(.plain)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 10)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                HStack(alignment: .bottom, spacing: 10) {
                    TextField("Ask to build something…", text: $model.composerText, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14))
                        .focused($composerFocused)
                        .lineLimit(1...8)
                        .padding(.vertical, 7)
                        .onSubmit { model.submit() }

                    Button(action: model.toggleVoiceInput) {
                        Image(systemName: microphoneSymbol)
                            .font(.system(size: 13.5, weight: .semibold))
                            .frame(width: 32, height: 32)
                            .contentShape(Circle())
                    }
                    .buttonStyle(SageCircleButtonStyle(
                        fill: isVoiceActive ? SageTheme.accent : SageTheme.hoverFill,
                        foreground: isVoiceActive ? .white : .secondary
                    ))
                    .help(isVoiceActive ? "Finish voice input" : "Start voice input")

                    if let task = selectedTask, !isFinished(task.status) {
                        Button {
                            model.cancel(taskID: task.taskID)
                        } label: {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 11.5, weight: .bold))
                                .frame(width: 32, height: 32)
                                .contentShape(Circle())
                        }
                        .buttonStyle(SageCircleButtonStyle(fill: SageTheme.warning, foreground: .white))
                        .help("Stop task")
                    } else {
                        Button {
                            model.submit()
                        } label: {
                            Image(systemName: "arrow.up")
                                .font(.system(size: 13.5, weight: .bold))
                                .frame(width: 32, height: 32)
                                .contentShape(Circle())
                        }
                        .buttonStyle(SageCircleButtonStyle(fill: SageTheme.accent, foreground: .white))
                        .disabled(!canSubmit)
                        .help("Send task")
                    }
                }

            }
            .padding(.horizontal, 16)
            .padding(.top, 13)
            .padding(.bottom, 11)
            .frame(maxWidth: 920)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 17, style: .continuous))
            .background(
                (composerFocused || composerHovering)
                    ? SageTheme.inputFocusedFill
                    : SageTheme.inputFill,
                in: RoundedRectangle(cornerRadius: 17, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 17, style: .continuous)
                    .stroke(
                        composerFocused
                            ? SageTheme.accent.opacity(0.3)
                            : (composerHovering ? SageTheme.strongStroke : SageTheme.stroke),
                        lineWidth: 1
                    )
            }
            .shadow(color: Color.black.opacity(0.14), radius: 18, y: 7)
            .onHover { composerHovering = $0 }
        }
        .padding(.horizontal, 32)
        .padding(.top, 6)
        .padding(.bottom, 22)
        .frame(maxWidth: .infinity)
    }

    private var selectedTask: Sage_Ipc_V1_TaskUpdate? {
        guard let id = model.selectedTaskID else { return nil }
        return model.tasks.first(where: { $0.taskID == id })
    }

    private var canSubmit: Bool {
        model.connectionState == .connected
            && !model.isSubmitting
            && !(selectedTask.map { !isFinished($0.status) } ?? false)
            && !model.composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var isVoiceActive: Bool {
        if case .listening = model.voiceState { return true }
        return false
    }

    private var microphoneSymbol: String {
        isVoiceActive ? "stop.fill" : "mic.fill"
    }

    private func statusLabel(_ status: Sage_Ipc_V1_TaskStatus) -> String {
        String(describing: status).replacingOccurrences(of: "_", with: " ").capitalized
    }

    private func statusColor(_ status: Sage_Ipc_V1_TaskStatus) -> Color {
        switch status {
        case .succeeded: return SageTheme.success
        case .failed, .interrupted: return SageTheme.danger
        case .cancelled: return .secondary
        case .waitingForApproval, .waitingForUser, .paused: return SageTheme.warning
        case .planning, .running, .pending: return SageTheme.accent
        default: return .secondary
        }
    }

    private func statusSymbol(_ status: Sage_Ipc_V1_TaskStatus) -> String {
        switch status {
        case .succeeded: return "checkmark.circle.fill"
        case .failed, .interrupted: return "exclamationmark.triangle.fill"
        case .cancelled: return "minus.circle"
        case .waitingForApproval: return "checkmark.shield"
        case .waitingForUser: return "questionmark.circle"
        case .paused: return "pause.circle"
        case .planning, .running: return "progress.indicator"
        default: return "circle"
        }
    }

    private func isFinished(_ status: Sage_Ipc_V1_TaskStatus) -> Bool {
        [.succeeded, .failed, .cancelled, .interrupted].contains(status)
    }

    private func eventIcon(_ kind: String) -> String {
        if kind.contains("failed") || kind.contains("denied") { return "exclamationmark.triangle" }
        if kind.contains("succeeded") || kind.contains("completed") { return "checkmark" }
        if kind.contains("approval") { return "checkmark.shield" }
        if kind.contains("observation") { return "eye" }
        return "sparkles"
    }

    private func questionSheet(_ question: Sage_Ipc_V1_QuestionRequest) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("More information needed")
                .font(.title2.weight(.semibold))
            Text(question.question)
            TextField("Answer", text: $questionAnswer, axis: .vertical)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Send") {
                    let answer = questionAnswer
                    questionAnswer = ""
                    model.answer(answer, question: question)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(questionAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(26)
        .frame(width: 480)
        .interactiveDismissDisabled()
    }
}

private struct ApprovalView: View {
    let approval: Sage_Ipc_V1_ApprovalRequest
    let approve: () -> Void
    let deny: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label(approval.title, systemImage: "exclamationmark.shield")
                .font(.title2.weight(.semibold))
            Text(approval.explanation)
            LabeledContent("Resource") {
                Text(approval.resource).textSelection(.enabled)
            }
            LabeledContent("Risk") {
                Text(String(describing: approval.risk).capitalized)
            }
            if approval.requiresNativeAuthentication {
                Label("macOS device authentication is required", systemImage: "touchid")
                    .foregroundStyle(.secondary)
            }
            if !approval.reversible {
                Label("Sage cannot promise this action can be undone", systemImage: "arrow.uturn.backward.slash")
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Deny", role: .cancel, action: deny)
                Spacer()
                Button("Approve once", action: approve)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(26)
        .frame(width: 520)
    }
}

enum SageTheme {
    static let canvas = Color(nsColor: .windowBackgroundColor)
    static let sidebar = Color(nsColor: .underPageBackgroundColor)
    static let card = Color(nsColor: .controlBackgroundColor).opacity(0.78)
    static let inputFill = Color.primary.opacity(0.045)
    static let inputFocusedFill = Color.primary.opacity(0.07)
    static let hoverFill = Color.primary.opacity(0.055)
    static let selectionFill = Color.primary.opacity(0.09)
    static let userBubble = Color.primary.opacity(0.075)
    static let stroke = Color.primary.opacity(0.075)
    static let strongStroke = Color.primary.opacity(0.13)
    static let accent = Color(red: 0.43, green: 0.48, blue: 0.98)
    static let success = Color(red: 0.28, green: 0.76, blue: 0.46)
    static let warning = Color(red: 0.93, green: 0.64, blue: 0.24)
    static let danger = Color(red: 0.92, green: 0.35, blue: 0.36)
}

private struct SageSidebarButtonStyle: ButtonStyle {
    var selected = false
    var externallyHovered = false

    func makeBody(configuration: Configuration) -> some View {
        SageSidebarButtonBody(
            configuration: configuration,
            selected: selected,
            externallyHovered: externallyHovered
        )
    }
}

private struct SageSidebarButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let selected: Bool
    let externallyHovered: Bool
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovering = false

    private var fill: Color {
        guard isEnabled else { return .clear }
        if configuration.isPressed { return SageTheme.selectionFill.opacity(1.35) }
        if selected { return SageTheme.selectionFill }
        if isHovering || externallyHovered { return SageTheme.hoverFill }
        return .clear
    }

    var body: some View {
        configuration.label
            .foregroundStyle(isEnabled ? Color.primary : Color.secondary)
            .background(fill, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .opacity(isEnabled ? 1 : 0.48)
            .onHover { isHovering = $0 }
            .animation(.easeOut(duration: 0.14), value: isHovering)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}

private struct SageCircleButtonStyle: ButtonStyle {
    let fill: Color
    let foreground: Color

    func makeBody(configuration: Configuration) -> some View {
        SageCircleButtonBody(configuration: configuration, fill: fill, foreground: foreground)
    }
}

private struct SageCircleButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let fill: Color
    let foreground: Color
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovering = false

    var body: some View {
        configuration.label
            .foregroundStyle(isEnabled ? foreground : Color.secondary.opacity(0.55))
            .background(
                isEnabled
                    ? (configuration.isPressed ? SageTheme.selectionFill : (isHovering ? fill.opacity(0.88) : fill))
                    : SageTheme.hoverFill,
                in: Circle()
            )
            .opacity(isEnabled ? 1 : 0.62)
            .onHover { isHovering = $0 }
            .animation(.easeOut(duration: 0.14), value: isHovering)
    }
}

struct SageBorderedButtonStyle: ButtonStyle {
    var destructive = false

    func makeBody(configuration: Configuration) -> some View {
        SageBorderedButtonBody(configuration: configuration, destructive: destructive)
    }
}

private struct SageBorderedButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let destructive: Bool
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovering = false

    var body: some View {
        configuration.label
            .font(.system(size: 12.5, weight: .medium))
            .foregroundStyle(destructive ? SageTheme.danger : Color.primary)
            .padding(.horizontal, 12)
            .frame(minHeight: 30)
            .background(
                configuration.isPressed
                    ? SageTheme.selectionFill
                    : (isHovering ? SageTheme.hoverFill : Color.clear),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(SageTheme.stroke, lineWidth: 1)
            }
            .opacity(isEnabled ? 1 : 0.45)
            .onHover { isHovering = $0 }
            .animation(.easeOut(duration: 0.14), value: isHovering)
    }
}

extension Sage_Ipc_V1_ApprovalRequest: Identifiable {
    var id: String { approvalID }
}

extension Sage_Ipc_V1_QuestionRequest: Identifiable {
    var id: String { questionID }
}
