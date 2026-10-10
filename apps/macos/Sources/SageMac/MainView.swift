import SwiftUI

struct MainView: View {
    @Bindable var model: AppModel
    @FocusState private var composerFocused: Bool
    @FocusState private var renameFocused: Bool
    @FocusState private var focusedTaskOptionsID: String?
    @FocusState private var searchFocused: Bool
    @State private var questionAnswer = ""
    @State private var composerHovering = false
    @State private var hoveredTaskID: String?
    @State private var searchText = ""
    @State private var activityExpanded = false
    @State private var followingLatest = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

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
        .onChange(of: model.composerText) { model.prepareIntent(model.composerText) }
        .onChange(of: model.taskFolders) { model.prepareIntent(model.composerText) }
        .onChange(of: model.pendingDecision?.id) { questionAnswer = "" }
        .onChange(of: model.selectedTaskID) {
            activityExpanded = false
            followingLatest = true
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
        .sheet(item: $model.pendingDecision) { decision in
            switch decision {
            case .approval(let approval):
                ApprovalView(
                    approval: approval,
                    resolving: model.resolvingDecision,
                    approve: { model.approve(approval) },
                    deny: { model.deny(approval) },
                    stop: { model.cancel(taskID: approval.taskID) },
                    later: { model.deferDecision() }
                )
                .interactiveDismissDisabled()
            case .question(let question):
                questionSheet(question)
            case .learning(let candidate):
                LearningApprovalView(
                    candidate: candidate,
                    busy: model.worldModelBusy,
                    sessionActive: model.learningSessionID != nil,
                    approve: { model.approveLearningCandidate(candidate) },
                    later: { model.deferDecision() }
                )
                .interactiveDismissDisabled()
            case .controllerDraft(let draft):
                ControllerReviewView(
                    draft: draft,
                    busy: model.worldModelBusy,
                    status: model.controllerDraftStatus,
                    review: { model.reviewControllerDraft(draft) },
                    run: { model.runReviewedController(draft) },
                    later: { model.deferDecision() }
                )
                .interactiveDismissDisabled()
            }
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

            Button(action: model.openGoalBuilder) {
                HStack(spacing: 10) {
                    Image(systemName: "target")
                        .font(.system(size: 13, weight: .medium))
                    Text("Goal builder")
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
            .sheet(isPresented: $model.goalBuilderPresented) {
                GoalBuilderView(model: model)
            }
            .disabled(model.connectionState != .connected
                      || model.worldModelBusy
                      || model.learningSessionID != nil
                      || model.pendingDecision != nil)
            .help("Compose a goal from current, reversibly tested application controls")
            .accessibilityLabel("Goal builder")
            .accessibilityHint("Preview a typed procedure. Starting it still requires fresh approval.")

            Button(action: model.openFileStreamCopy) {
                HStack(spacing: 10) {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 13, weight: .medium))
                    Text("Copy a file")
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
            .sheet(isPresented: $model.fileStreamCopyPresented) {
                FileStreamCopyView(model: model)
            }
            .disabled(model.connectionState != .connected
                      || model.worldModelBusy
                      || model.learningSessionID != nil
                      || model.pendingDecision != nil)
            .help("Stream one local file through Sage's bounded, approved native file path")
            .accessibilityLabel("Copy a file")
            .accessibilityHint("Choose a source and destination. Sage asks before reading or writing.")

            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search chats", text: $searchText)
                    .textFieldStyle(.plain)
                    .focused($searchFocused)
                    .accessibilityLabel("Search chats")
                    .help("Search chats (⌘F)")
                if !searchText.isEmpty {
                    Button { searchText = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Clear chat search")
                }
            }
            .font(.system(size: 12))
            .padding(9)
            .background(SageTheme.inputFill, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(searchFocused ? SageTheme.accent.opacity(0.5) : SageTheme.stroke))
            .padding(.horizontal, 12)
            .padding(.top, 12)
            .padding(.bottom, 14)
            .background {
                Button("Search chats") { searchFocused = true }
                    .keyboardShortcut("f", modifiers: [.command])
                    .hidden().accessibilityHidden(true)
            }

            ScrollView(.vertical) {
                LazyVStack(spacing: 2) {
                    if filteredTasks.isEmpty {
                        Text(searchText.isEmpty ? "No chats yet" : "No matching chats")
                            .font(.system(size: 12))
                            .foregroundStyle(.tertiary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 8)
                    } else {
                        if !pinnedTasks.isEmpty {
                            sidebarSection("Pinned")
                            ForEach(pinnedTasks, id: \.taskID) { task in taskRow(task) }
                        }
                        if !recentTasks.isEmpty {
                            sidebarSection(searchText.isEmpty ? "Recent" : "Results")
                            ForEach(recentTasks, id: \.taskID) { task in taskRow(task) }
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

            if model.decisionCount > 0 {
                Button(action: model.reviewDecisions) {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.shield")
                        Text("Needs your review")
                        Spacer(minLength: 0)
                        Text("\(model.decisionCount)").monospacedDigit()
                    }
                    .font(.system(size: 12, weight: .medium))
                    .padding(10)
                    .background(SageTheme.warning.opacity(0.10), in: RoundedRectangle(cornerRadius: 9))
                }
                .buttonStyle(.plain)
                .foregroundStyle(SageTheme.warning)
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
                .accessibilityLabel("Review \(model.decisionCount) pending requests")
            }

            connectionIndicator

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

    private var filteredTasks: [Sage_Ipc_V2_TaskUpdate] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return model.visibleTasks.filter {
            query.isEmpty || model.displayTitle(for: $0).localizedCaseInsensitiveContains(query)
                || $0.request.localizedCaseInsensitiveContains(query)
        }
    }

    private var pinnedTasks: [Sage_Ipc_V2_TaskUpdate] { filteredTasks.filter { model.isPinned($0.taskID) } }
    private var recentTasks: [Sage_Ipc_V2_TaskUpdate] { filteredTasks.filter { !model.isPinned($0.taskID) } }

    private func sidebarSection(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 10.5, weight: .semibold))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.top, 6)
            .padding(.bottom, 6)
    }

    private var connectionIndicator: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(model.connectionState == .connected ? SageTheme.success : SageTheme.warning)
                .frame(width: 6, height: 6)
                .accessibilityHidden(true)
            Text(model.connectionState == .connected ? "Connected" : model.connectionState == .starting ? "Connecting…" : "Disconnected")
                .font(.system(size: 10.5, weight: .medium))
            Spacer(minLength: 0)
            if case .failed = model.connectionState {
                Button("Reconnect", action: model.reconnect).buttonStyle(.plain)
                    .font(.system(size: 10.5, weight: .semibold))
            }
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 18)
        .padding(.bottom, 12)
        .help(model.connectionState.label)
    }

    private func taskRow(_ task: Sage_Ipc_V2_TaskUpdate) -> some View {
        let selected = model.selectedTaskID == task.taskID && !model.settingsVisible
        let optionsFocused = focusedTaskOptionsID == task.taskID
        let showingOptions = hoveredTaskID == task.taskID || optionsFocused

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
                                    .accessibilityHidden(true)
                            }
                            Text(model.displayTitle(for: task))
                                .font(.system(size: 12.5, weight: .medium))
                                .lineLimit(1)
                                .truncationMode(.tail)
                            Spacer(minLength: 0)
                            if !isFinished(task.status) || TaskPresentation.needsAttention(task) {
                                Image(systemName: statusSymbol(task.status))
                                    .font(.system(size: 10))
                                    .foregroundStyle(statusColor(task.status))
                                    .help(statusLabel(task.status))
                            }
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
                    .accessibilityLabel(
                        model.isPinned(task.taskID)
                            ? "\(model.displayTitle(for: task)), pinned"
                            : model.displayTitle(for: task)
                    )
                    .accessibilityHint("Open chat")
                    .accessibilityValue(statusLabel(task.status))

                    taskOptionsMenu(task, isFocused: optionsFocused, isVisible: showingOptions)
                        .padding(.trailing, 5)
                        .opacity(showingOptions ? 1 : 0)
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

    private func taskOptionsMenu(
        _ task: Sage_Ipc_V2_TaskUpdate,
        isFocused: Bool,
        isVisible: Bool
    ) -> some View {
        Menu {
            taskMenuItems(task)
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(isVisible ? .primary : .secondary)
                .frame(width: 32, height: 32)
                .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .background(
                    isFocused ? SageTheme.selectionFill : (isVisible ? SageTheme.hoverFill : Color.clear),
                    in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .stroke(isFocused ? SageTheme.accent.opacity(0.55) : Color.clear, lineWidth: 1)
                }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .focused($focusedTaskOptionsID, equals: task.taskID)
        .help("More chat actions")
        .accessibilityLabel("More actions for \(model.displayTitle(for: task))")
        .accessibilityHint("Rename, pin, or delete this chat")
    }

    @ViewBuilder
    private func taskMenuItems(_ task: Sage_Ipc_V2_TaskUpdate) -> some View {
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

    private func taskToolbar(_ task: Sage_Ipc_V2_TaskUpdate) -> some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayTitle(for: task))
                    .font(.system(size: 15.5, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 16)
            if task.undoState.isEmpty && task.continuedTaskID.isEmpty && (task.status == .interrupted || task.status == .paused) {
                Button("Resume") { model.resume(taskID: task.taskID) }.buttonStyle(SageBorderedButtonStyle())
            }
            if task.undoState.isEmpty && task.status == .succeeded && task.totalActions > 0 {
                Button("Save skill") { model.workflow("capture_skill", id: task.taskID, name: model.displayTitle(for: task)) }.buttonStyle(SageBorderedButtonStyle())
                Button("Save controller draft") { model.compileControllerDraft(taskID: task.taskID) }
                    .buttonStyle(SageBorderedButtonStyle())
                    .disabled(model.worldModelBusy)
                    .help("Compile an unreviewed draft from a fully verified UI procedure. It grants no access and still requires fresh review.")
            }
            if task.undoAvailable {
                Button {
                    model.undo(taskID: task.taskID)
                } label: {
                    Label(task.undoState == "dispatched" || task.undoState == "uncertain" ? "Check Undo" : task.undoState == "prepared" ? "Retry Undo" : "Undo", systemImage: "arrow.uturn.backward")
                }
                .buttonStyle(SageBorderedButtonStyle())
                .help(task.undoSummary.isEmpty ? "Undo the last reversible action" : task.undoSummary)
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
                ScrollViewReader { proxy in
                  ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(model.messages) { message in
                            HStack {
                                if message.role == "user" { Spacer(minLength: 48) }
                                Text(message.content)
                                    .font(.system(size: 14))
                                    .textSelection(.enabled)
                                    .padding(12)
                                    .background(message.role == "user" ? SageTheme.userBubble : Color.clear, in: RoundedRectangle(cornerRadius: 12))
                                if message.role != "user" { Spacer(minLength: 48) }
                            }.padding(.bottom, 12)
                        }
                        if !model.messages.contains(where: { $0.taskId == task.taskID && $0.role == "user" }) {
                            requestMessage(task)
                        }
                        if let response = TaskPresentation.response(for: task, messages: model.messages, streamed: model.streamedResponses[task.taskID]) {
                            HStack(alignment: .top, spacing: 12) {
                                SageLogo(size: 22).accessibilityHidden(true)
                                Text(response)
                                    .font(.system(size: 14))
                                    .textSelection(.enabled)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(.vertical, 16)
                        }
                        taskStatusCard(task)
                        if !task.actions.isEmpty || !model.timeline.isEmpty {
                            DisclosureGroup(isExpanded: $activityExpanded) {
                                ForEach(task.actions, id: \.actionID) { action in
                                    HStack(alignment: .top) {
                                        Text(action.summary).textSelection(.enabled)
                                        Spacer(minLength: 12)
                                        Text(action.status).foregroundStyle(.secondary)
                                    }.font(.system(size: 12)).padding(.vertical, 6)
                                }
                                ForEach(Array(model.timeline.reversed().enumerated()), id: \.offset) { _, event in
                                    timelineEvent(event)
                                }
                            } label: {
                                Text("Activity").font(.system(size: 12, weight: .medium))
                            }
                            .padding(.top, 16)
                            .tint(.secondary)
                        }
                        Color.clear.frame(height: 1).id("conversation-bottom")
                    }
                    .frame(maxWidth: 920, alignment: .leading)
                    .padding(.horizontal, 32)
                    .padding(.top, 12)
                    .padding(.bottom, 18)
                    .frame(maxWidth: .infinity)
                  }
                  .scrollIndicators(.automatic)
                  .simultaneousGesture(DragGesture(minimumDistance: 3).onChanged { _ in followingLatest = false })
                  .onChange(of: model.streamedResponses[task.taskID]) {
                      if followingLatest { proxy.scrollTo("conversation-bottom", anchor: .bottom) }
                  }
                  .onChange(of: model.messages.count) {
                      if followingLatest { proxy.scrollTo("conversation-bottom", anchor: .bottom) }
                  }
                  .overlay(alignment: .bottomTrailing) {
                      if !followingLatest {
                          Button {
                              followingLatest = true
                              proxy.scrollTo("conversation-bottom", anchor: .bottom)
                          } label: { Label("Latest", systemImage: "arrow.down") }
                          .buttonStyle(SageBorderedButtonStyle())
                          .padding(16)
                      }
                  }
                }
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func requestMessage(_ task: Sage_Ipc_V2_TaskUpdate) -> some View {
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

    private func taskStatusCard(_ task: Sage_Ipc_V2_TaskUpdate) -> some View {
        let continued = !task.continuedTaskID.isEmpty && task.status != .cancelled
        return VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label(continued ? "Continued" : statusLabel(task.status), systemImage: continued ? "arrow.forward" : statusSymbol(task.status))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(continued ? Color.secondary : statusColor(task.status))
                Spacer(minLength: 12)
            if task.totalActions > 0 {
                    Text("\(TaskPresentation.verifiedCount(task)) of \(task.totalActions) actions verified")
                        .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
                }
            }
            if !isFinished(task.status), !task.currentAction.isEmpty {
                Text(task.currentAction).font(.system(size: 12)).foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            if !task.intentChangeSummary.isEmpty {
                Label(task.intentChangeSummary, systemImage: "arrow.triangle.branch")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !task.routineSummary.isEmpty {
                Label(task.routineSummary, systemImage: "sparkles")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if model.controllerDraftTaskID == task.taskID, !model.controllerDraftStatus.isEmpty {
                Text(model.controllerDraftStatus)
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if !isFinished(task.status), !task.actions.isEmpty {
                HStack(spacing: 12) {
                    executionStage("Preparing", symbol: "gearshape", count: task.actions.filter { $0.status == "compiling" }.count)
                    executionStage("Running", symbol: "play.fill", count: task.actions.filter { $0.status == "running" }.count)
                    executionStage("Verifying", symbol: "checkmark.shield", count: task.actions.filter { $0.status == "verifying" }.count)
                }
            }
            if TaskPresentation.needsAttention(task), !task.summary.isEmpty {
                Text(task.summary).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if task.hasExecutionFacts && task.executionFacts.uncertain > 0 {
                Label("Some effects still need verification", systemImage: "exclamationmark.triangle")
                    .font(.system(size: 12)).foregroundStyle(SageTheme.warning)
            }
            if model.stoppingTaskIDs.contains(task.taskID) {
                Text("Stopping…").font(.system(size: 12)).foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
    }

    private func executionStage(_ label: String, symbol: String, count: Int) -> some View {
        Label("\(label) \(count)", systemImage: symbol)
            .font(.system(size: 11, weight: count > 0 ? .semibold : .regular))
            .foregroundStyle(count > 0 ? SageTheme.accent : Color.secondary)
            .monospacedDigit()
            .accessibilityLabel("\(count) actions \(label.lowercased())")
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            SageLogo(size: 46)
                .shadow(color: Color.black.opacity(0.16), radius: 12, y: 6)
            Text("What would you like to do?")
                .font(.system(size: 24, weight: .semibold))
            HStack(spacing: 10) {
                Button("Explore a folder") {
                    model.composerText = "List files"
                    model.chooseTaskFolder()
                    composerFocused = true
                }.buttonStyle(SageBorderedButtonStyle())
                Button("Ask a question") {
                    model.composerText = "Help me understand "
                    composerFocused = true
                }.buttonStyle(SageBorderedButtonStyle())
            }.padding(.top, 6)
        }
        .padding(.horizontal, 32)
        .padding(.bottom, 64)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func timelineEvent(_ event: Sage_Ipc_V2_AgentEvent) -> some View {
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
            if let notice = model.refreshNotice {
                HStack {
                    Text(notice).font(.system(size: 12)).foregroundStyle(SageTheme.warning)
                    Spacer(minLength: 8)
                    Button("Reconnect", action: model.reconnect).buttonStyle(.plain)
                }.frame(maxWidth: 920)
            }
            if model.connectionState == .connected && !model.storageLocked {
                HStack {
                    Text("Basic commands run locally. Sage's first-party model engine is in development.").font(.system(size: 12)).foregroundStyle(.secondary)
                    Spacer(minLength: 8)
                    Button("Details") { model.settingsVisible = true }.buttonStyle(.plain)
                }.frame(maxWidth: 920)
            }
            if model.storageLocked {
                Button("Open protected history") { model.unlockHistory() }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
            }
            if let preview = model.intentPreview, !preview.steps.isEmpty || !preview.routineSuggestions.isEmpty || preview.status == "paused" || preview.status == "stopping" {
                let streamingRead = !preview.streamedPrefix.isEmpty
                    && preview.steps.first?.lowercased().hasPrefix("read ") == true
                VStack(alignment: .leading, spacing: 8) {
                    Label(preview.status == "unavailable" ? "Action unavailable" : preview.status == "paused" ? "Paused for your correction" : preview.status == "stopping" ? "Stopping" : preview.status == "suggested" ? "Possible learned path" : streamingRead ? "Reading while you speak" : !preview.streamedPrefix.isEmpty ? "Ready for approval while you speak" : preview.status == "prepared" ? "Ready to open" : preview.status == "preparing" ? "Preparing your request" : "Ready on this device",
                          systemImage: preview.status == "paused" ? "pause.circle" : preview.status == "suggested" ? "arrow.triangle.branch" : streamingRead ? "doc.text" : !preview.streamedPrefix.isEmpty ? "waveform" : "bolt.circle")
                        .font(.system(size: 12, weight: .semibold)).foregroundStyle(SageTheme.accent)
                    ForEach(Array(preview.steps.enumerated()), id: \.offset) { index, step in
                        HStack(spacing: 8) {
                            Text("\(index + 1)").monospacedDigit().foregroundStyle(.secondary)
                            Text(step).lineLimit(2)
                        }.font(.system(size: 12))
                    }
                    if !preview.detail.isEmpty { Text(preview.detail).font(.system(size: 11)).foregroundStyle(.secondary) }
                    if !preview.routineSuggestions.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            Label("Possible learned paths", systemImage: "arrow.triangle.branch")
                                .font(.system(size: 11.5, weight: .semibold))
                                .foregroundStyle(SageTheme.accent)
                            Text(preview.routineSuggestionDetail)
                                .font(.system(size: 11)).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            ScrollView(.horizontal) {
                                HStack(spacing: 8) {
                                    ForEach(Array(preview.routineSuggestions.enumerated()), id: \.offset) { _, request in
                                        Button {
                                            model.composerText = request
                                            model.composerFocusToken = UUID()
                                        } label: {
                                            Text("Use: \(request)")
                                                .lineLimit(2)
                                                .fixedSize(horizontal: false, vertical: true)
                                                .frame(maxWidth: 260, alignment: .leading)
                                        }
                                        .buttonStyle(SageBorderedButtonStyle())
                                        .accessibilityLabel("Use learned request: \(request)")
                                    }
                                }
                            }
                            .scrollIndicators(.hidden)
                            Text("Choosing a path fills the message box. Send starts it.")
                                .font(.system(size: 10.5)).foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .background(SageTheme.stroke.opacity(0.28), in: RoundedRectangle(cornerRadius: 9))
                    }
                }
                .frame(maxWidth: 888, alignment: .leading).padding(14)
                .background(SageTheme.hoverFill, in: RoundedRectangle(cornerRadius: 12))
                .accessibilityElement(children: preview.routineSuggestions.isEmpty ? .combine : .contain)
            }
            if model.isSpeaking {
                HStack(spacing: 8) {
                    Label("Speaking", systemImage: "speaker.wave.2.fill")
                        .symbolEffect(.variableColor.iterative)
                        .accessibilityAddTraits(.updatesFrequently)
                    Spacer(minLength: 0)
                    Button {
                        model.stopSpokenReply()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .font(.system(size: 11.5, weight: .medium))
                    }
                    .buttonStyle(.plain)
                    .help("Stop spoken reply")
                    .accessibilityLabel("Stop spoken reply")
                }
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(maxWidth: 920)
                .padding(.bottom, 2)
                .accessibilityElement(children: .contain)
            }
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

                if !model.taskFolders.isEmpty {
                    HStack {
                        Text("Can read: " + model.taskFolders.map { URL(fileURLWithPath: $0).lastPathComponent }.joined(separator: ", "))
                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        Button("Clear") { model.taskFolders = [] }.buttonStyle(.plain)
                    }.padding(.bottom, 8)
                }
                HStack(alignment: .bottom, spacing: 10) {
                    Button(action: model.chooseTaskFolder) {
                        Image(systemName: "folder.badge.plus")
                            .frame(width: 32, height: 32)
                            .contentShape(Circle())
                    }
                    .buttonStyle(.plain)
                    .help("Choose folders this task can read")
                    .accessibilityLabel("Choose folders")
                    .accessibilityHint("Allow this task to read files in selected folders")
                    TextField("Ask Sage anything…", text: $model.composerText, axis: .vertical)
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
                    .accessibilityLabel(isVoiceActive ? "Finish voice input" : "Start voice input")

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
                        .accessibilityLabel("Stop task")
                    }
                    if (selectedTask.map({ isFinished($0.status) }) ?? true) || !model.composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
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
                        .help(selectedTask.map { !isFinished($0.status) } == true ? "Update request and stop the previous plan" : "Send")
                        .accessibilityLabel(selectedTask.map { !isFinished($0.status) } == true ? "Update request" : "Send")
                        .accessibilityHint("Submit your current request")
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

    private var selectedTask: Sage_Ipc_V2_TaskUpdate? {
        guard let id = model.selectedTaskID else { return nil }
        return model.tasks.first(where: { $0.taskID == id })
    }

    private var canSubmit: Bool {
        model.connectionState == .connected
            && !model.isSubmitting
            && !model.composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var isVoiceActive: Bool {
        if case .listening = model.voiceState { return true }
        return false
    }

    private var microphoneSymbol: String {
        isVoiceActive ? "stop.fill" : "mic.fill"
    }

    private func statusLabel(_ status: Sage_Ipc_V2_TaskStatus) -> String {
        TaskPresentation.label(status)
    }

    private func statusColor(_ status: Sage_Ipc_V2_TaskStatus) -> Color {
        switch status {
        case .succeeded: return SageTheme.success
        case .failed, .interrupted: return SageTheme.danger
        case .cancelled: return .secondary
        case .waitingForApproval, .waitingForUser, .paused, .partial: return SageTheme.warning
        case .planning, .running, .pending: return SageTheme.accent
        default: return .secondary
        }
    }

    private func statusSymbol(_ status: Sage_Ipc_V2_TaskStatus) -> String {
        switch status {
        case .succeeded: return "checkmark.circle.fill"
        case .answered: return "text.bubble"
        case .partial: return "exclamationmark.circle"
        case .failed, .interrupted: return "exclamationmark.triangle.fill"
        case .cancelled: return "minus.circle"
        case .waitingForApproval: return "checkmark.shield"
        case .waitingForUser: return "questionmark.circle"
        case .paused: return "pause.circle"
        case .planning, .running: return "progress.indicator"
        default: return "circle"
        }
    }

    private func isFinished(_ status: Sage_Ipc_V2_TaskStatus) -> Bool {
        [.succeeded, .answered, .partial, .failed, .cancelled, .interrupted].contains(status)
    }

    private func eventIcon(_ kind: String) -> String {
        if kind.contains("reference") { return "text.viewfinder" }
        if kind.contains("failed") || kind.contains("denied") { return "exclamationmark.triangle" }
        if kind.contains("succeeded") || kind.contains("completed") { return "checkmark" }
        if kind.contains("approval") { return "checkmark.shield" }
        if kind.contains("observation") { return "eye" }
        return "sparkles"
    }

    private func questionSheet(_ question: Sage_Ipc_V2_QuestionRequest) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("More information needed")
                .font(.title2.weight(.semibold))
            Text(question.question)
            TextField("Answer", text: $questionAnswer, axis: .vertical)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button("Stop task", role: .destructive) { model.cancel(taskID: question.taskID) }
                Button("Later", action: model.deferDecision)
                    .disabled(model.resolvingDecision)
                Spacer()
                Button("Send") {
                    model.answer(questionAnswer, question: question)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(model.resolvingDecision || questionAnswer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(26)
        .frame(width: 480)
        .interactiveDismissDisabled()
    }
}

private struct LearningApprovalView: View {
    let candidate: LearningCandidate
    let busy: Bool
    let sessionActive: Bool
    let approve: () -> Void
    let later: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Label("Sage found a control to learn", systemImage: "slider.horizontal.3")
                .font(.system(size: 17, weight: .semibold))
            VStack(alignment: .leading, spacing: 6) {
                Text(candidate.systemLabel)
                    .font(.system(size: 14, weight: .medium))
                Text("\(candidate.label) (\(candidate.role))")
                    .font(.system(size: 13))
            }
            Text("This temporary approval covers this control only, for up to 20 experiments or 10 minutes. Sage will ask you to start each test, change the value by at most one unit, restore its exact previous value and verify the result. No other app action is authorized.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if sessionActive {
                Text("Stop the current learning session from Sage’s menu bar before approving another control.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(SageTheme.warning)
            }
            HStack {
                Button("Review later", action: later)
                    .disabled(busy)
                Spacer()
                Button("Approve temporary session", action: approve)
                    .buttonStyle(.borderedProminent)
                    .disabled(busy || sessionActive)
            }
        }
        .padding(24)
        .frame(width: 500)
    }
}

private struct ApprovalView: View {
    let approval: Sage_Ipc_V2_ApprovalRequest
    let resolving: Bool
    let approve: () -> Void
    let deny: () -> Void
    let stop: () -> Void
    let later: () -> Void

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
                Button("Stop task", role: .destructive, action: stop)
                Button("Later", action: later).disabled(resolving)
                Button("Deny", role: .cancel, action: deny)
                    .disabled(resolving)
                Spacer()
                Button("Approve once", action: approve)
                    .disabled(resolving)
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

extension Sage_Ipc_V2_ApprovalRequest: Identifiable {
    var id: String { approvalID }
}

extension Sage_Ipc_V2_QuestionRequest: Identifiable {
    var id: String { questionID }
}
