import AppKit
import AVFoundation
import Foundation
import Observation
import Speech

@MainActor
@Observable
final class AppModel {
    private struct TaskMetadata: Codable {
        var title: String?
        var pinned = false
        var deleted = false
    }

    enum ConnectionState: Equatable {
        case starting
        case connected
        case failed(String)

        var label: String {
            switch self {
            case .starting: "Starting local core…"
            case .connected: "Local core connected"
            case .failed(let message): message
            }
        }
    }

    var connectionState: ConnectionState = .starting
    var tasks: [Sage_Ipc_V2_TaskUpdate] = []
    var timeline: [Sage_Ipc_V2_AgentEvent] = []
    var pendingApproval: Sage_Ipc_V2_ApprovalRequest?
    var pendingQuestion: Sage_Ipc_V2_QuestionRequest?
    var composerText = ""
    var taskFolders: [String] = []
    var storageLocked = true
    var selectedTaskID: String?
    var selectedConversationID: String?
    var conversations: [ConversationRecord] = []
    var messages: [ConversationMessage] = []
    var memories: [MemoryRecord] = []
    var memoryEnabled = true
    var skills: [SkillRecord] = []
    var workflows: [WorkflowRecord] = []
    var schedules: [ScheduleRecord] = []
    var settingsVisible = false
    var composerFocusToken = UUID()
    var renameFocusToken = UUID()
    var editingTaskID: String?
    var editingTaskTitle = ""
    var deleteCandidateID: String?
    var deleteCandidateTitle = ""
    var errorMessage: String?
    var voiceState: VoiceInputController.State = .idle
    var voiceTranscript = ""
    var voiceNotice: String?
    var wakeWordEnabled = UserDefaults.standard.object(forKey: "wakeWordEnabled") as? Bool ?? false
    var wakePhrase = UserDefaults.standard.string(forKey: "wakePhrase") ?? "Hey Sage"
    var providerSaveMessage: String?
    var providerSaving = false
    var providerTestMessage: String?
    var providerTesting = false
    var providerSettings: Sage_Ipc_V2_ProviderSettings?
    private(set) var isSubmitting = false
    private(set) var draftActive = false

    private let supervisor = CoreSupervisor()
    private let client = SageCoreClient()
    private let platformAdapter = PlatformAdapter()
    private let voiceOverlay = OverlayWindowController()
    private let voiceInput = VoiceInputController()
    private var taskMetadata: [String: TaskMetadata] = [:]
    private var timelinesByTaskID: [String: [Sage_Ipc_V2_AgentEvent]] = [:]
    private var lastSubmittedRequest = ""
    private var lastSubmittedAt: Date?
    private var started = false

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.taskMetadataKey),
           let saved = try? JSONDecoder().decode([String: TaskMetadata].self, from: data) {
            taskMetadata = saved
        }
    }

    private static let taskMetadataKey = "taskMetadata.v1"

    var visibleTasks: [Sage_Ipc_V2_TaskUpdate] {
        var seen = Set<String>()
        let visible = Set(conversations.map(\.id))
        return tasks.filter { task in
            let id = task.conversationID.isEmpty ? task.taskID : task.conversationID
            return (conversations.isEmpty || visible.contains(id)) && seen.insert(id).inserted
        }
            .enumerated()
            .filter { !((taskMetadata[$0.element.taskID]?.deleted) ?? false) }
            .sorted { lhs, rhs in
                let lhsPinned = taskMetadata[lhs.element.taskID]?.pinned ?? false
                let rhsPinned = taskMetadata[rhs.element.taskID]?.pinned ?? false
                if lhsPinned != rhsPinned { return lhsPinned }
                return lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    func displayTitle(for task: Sage_Ipc_V2_TaskUpdate) -> String {
        if let conversation = conversations.first(where: { $0.id == task.conversationID }) { return conversation.title }
        let title = taskMetadata[task.taskID]?.title?.trimmingCharacters(in: .whitespacesAndNewlines)
        return title?.isEmpty == false ? title! : task.request
    }

    func isPinned(_ taskID: String) -> Bool {
        if let task = tasks.first(where: { $0.taskID == taskID }), let conversation = conversations.first(where: { $0.id == task.conversationID }) { return conversation.pinned }
        return taskMetadata[taskID]?.pinned ?? false
    }

    func selectTask(_ taskID: String) {
        guard tasks.contains(where: { $0.taskID == taskID }) else { return }
        selectedTaskID = taskID
        selectedConversationID = tasks.first(where: { $0.taskID == taskID })?.conversationID
        messages = []
        knowledge("list")
        settingsVisible = false
        draftActive = false
        editingTaskID = nil
        timeline = timelinesByTaskID[taskID] ?? []
        composerText = ""
    }

    func beginRename(_ task: Sage_Ipc_V2_TaskUpdate) {
        settingsVisible = false
        editingTaskID = task.taskID
        editingTaskTitle = displayTitle(for: task)
        renameFocusToken = UUID()
    }

    func commitRename() {
        guard let taskID = editingTaskID else { return }
        let title = editingTaskTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else {
            editingTaskID = nil
            return
        }
        guard let task = tasks.first(where: { $0.taskID == taskID }) else {
            editingTaskID = nil
            return
        }
        if title == task.request {
            taskMetadata[taskID]?.title = nil
        } else {
            var metadata = taskMetadata[taskID] ?? TaskMetadata()
            metadata.title = title
            taskMetadata[taskID] = metadata
        }
        persistTaskMetadata()
        if let task = tasks.first(where: { $0.taskID == taskID }) { updateConversation(task, title: title) }
        editingTaskID = nil
    }

    func cancelRename() {
        editingTaskID = nil
    }

    func togglePinned(_ taskID: String) {
        if let task = tasks.first(where: { $0.taskID == taskID }) { updateConversation(task, pinned: !isPinned(taskID)); return }
        var metadata = taskMetadata[taskID] ?? TaskMetadata()
        metadata.pinned.toggle()
        taskMetadata[taskID] = metadata
        persistTaskMetadata()
    }

    func requestDelete(_ task: Sage_Ipc_V2_TaskUpdate) {
        editingTaskID = nil
        deleteCandidateID = task.taskID
        deleteCandidateTitle = displayTitle(for: task)
    }

    func cancelDelete() {
        deleteCandidateID = nil
        deleteCandidateTitle = ""
    }

    func confirmDelete() {
        guard let taskID = deleteCandidateID else { return }
        if let task = tasks.first(where: { $0.taskID == taskID }) { updateConversation(task, archived: true) }
        if let task = tasks.first(where: { $0.taskID == taskID }), !isFinished(task.status) {
            cancel(taskID: taskID)
        }
        var metadata = taskMetadata[taskID] ?? TaskMetadata()
        metadata.deleted = true
        taskMetadata[taskID] = metadata
        persistTaskMetadata()
        if selectedTaskID == taskID {
            selectedTaskID = nil
            draftActive = true
            timeline = []
            composerText = ""
            focusComposer()
        }
        cancelDelete()
    }

    func start() async {
        guard !started else { return }
        started = true
        configureVoiceInput()
        client.onAdapterRequest = { [weak self] request in
            guard let self else { return Sage_Ipc_V2_AdapterResult() }
            return await self.platformAdapter.handle(request)
        }
        do {
            let secret = try IPCSecretStore().loadOrCreateSecret()
            do {
                try await client.connect()
            } catch {
                try supervisor.startIfNeeded(secret: secret)
                try await connectWithRetry()
            }
            connectionState = .connected
            client.onEvent = { [weak self] event in
                Task { @MainActor in
                    self?.consume(event)
                }
            }
            try await client.requestState(includeCompleted: true)
            voiceInput.configureWakeWord(enabled: wakeWordEnabled, phrase: wakePhrase)
        } catch {
            connectionState = .failed(error.localizedDescription)
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        voiceInput.stop()
        voiceOverlay.hide()
        client.disconnect()
        supervisor.detach()
    }

    func submit(source: Sage_Ipc_V2_InputSource = .typed) {
        let request = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty, !isSubmitting, !selectedTaskIsActive else { return }
        if request == lastSubmittedRequest,
           let lastSubmittedAt,
           Date().timeIntervalSince(lastSubmittedAt) < 0.6 {
            return
        }
        lastSubmittedRequest = request
        lastSubmittedAt = Date()
        isSubmitting = true
        if selectedConversationID == nil { selectedConversationID = UUID().uuidString.lowercased() }
        draftActive = false
        composerText = ""
        Task {
            do {
                try await client.submitTask(request, source: source, conversationID: selectedConversationID ?? "", folders: taskFolders)
                taskFolders = []
                if source == .voice {
                    voiceInteractionExecuting(request)
                }
            } catch {
                isSubmitting = false
                draftActive = true
                errorMessage = error.localizedDescription
            }
        }
    }

    func unlockHistory() {
        Task { do { try await client.unlockStorage() } catch { errorMessage = error.localizedDescription } }
    }

    func chooseTaskFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.message = "Allow this task to read files in the selected folders. Changes will still need approval."
        panel.prompt = "Allow reading"
        if panel.runModal() == .OK { taskFolders = panel.urls.map(\.path) }
    }

    func approve(_ approval: Sage_Ipc_V2_ApprovalRequest) {
        Task {
            do {
                let authenticated: Bool
                if approval.requiresNativeAuthentication {
                    authenticated = try await NativeAuthentication.authenticate(
                        reason: approval.explanation
                    )
                } else {
                    authenticated = false
                }
                try await client.resolveApproval(
                    approval,
                    approve: true,
                    nativeAuthenticationSatisfied: authenticated
                )
                pendingApproval = nil
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func deny(_ approval: Sage_Ipc_V2_ApprovalRequest) {
        Task {
            do {
                try await client.resolveApproval(
                    approval,
                    approve: false,
                    nativeAuthenticationSatisfied: false
                )
                pendingApproval = nil
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func answer(_ answer: String, question: Sage_Ipc_V2_QuestionRequest) {
        Task {
            do {
                try await client.answer(question, text: answer)
                pendingQuestion = nil
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func cancel(taskID: String) {
        Task {
            do {
                try await client.control(taskID: taskID, operation: .cancel)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func undo(taskID: String) {
        Task {
            do {
                try await client.undo(taskID: taskID)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func focusComposer() {
        settingsVisible = false
        draftActive = selectedTaskID == nil
        composerFocusToken = UUID()
    }

    func newTask() {
        if draftActive && selectedTaskID == nil {
            focusComposer()
            return
        }
        selectedTaskID = nil
        selectedConversationID = nil
        messages = []
        timeline = []
        composerText = ""
        editingTaskID = nil
        draftActive = true
        focusComposer()
    }

    func refreshState() {
        Task {
            do {
                try await client.requestState(includeCompleted: true)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func applyKnowledge(_ json: String) {
        guard let data = try? decodeKnowledge(KnowledgeData.self, json) else { return }
        conversations = data.conversations; memories = data.memories; memoryEnabled = data.memoryEnabled
        let incoming = data.messages.filter { $0.conversationId == selectedConversationID }
        if !incoming.isEmpty { messages = incoming }
    }

    func knowledge(_ operation: String, id: String = "", content: String = "", enabled: Bool = true) {
        var request = Sage_Ipc_V2_KnowledgeCommand()
        request.operation = operation; request.id = id; request.content = content; request.enabled = enabled
        request.conversationID = selectedConversationID ?? ""
        Task { do { try await client.knowledge(request) } catch { errorMessage = error.localizedDescription } }
    }

    private func updateConversation(_ task: Sage_Ipc_V2_TaskUpdate, title: String? = nil, pinned: Bool? = nil, archived: Bool = false) {
        var request = Sage_Ipc_V2_KnowledgeCommand()
        request.operation = "conversation"; request.id = task.conversationID
        request.content = title ?? displayTitle(for: task); request.pinned = pinned ?? isPinned(task.taskID); request.archived = archived
        request.conversationID = selectedConversationID ?? ""
        Task {
            do {
                if archived && !isFinished(task.status) { try await client.control(taskID: task.taskID, operation: .cancel) }
                try await client.knowledge(request)
                try await client.requestState(includeCompleted: true)
            } catch { errorMessage = error.localizedDescription }
        }
    }

    func workflow(_ operation: String, id: String = "", name: String = "", json: String = "") {
        var request = Sage_Ipc_V2_WorkflowCommand()
        request.operation = operation; request.id = id; request.name = name; request.json = json
        request.conversationID = selectedConversationID ?? ""
        Task { do { try await client.workflow(request) } catch { errorMessage = error.localizedDescription } }
    }

    func schedule(name: String, request: String, date: Date, interval: Int, folder: String) {
        let formatter = ISO8601DateFormatter()
        let trigger: [String: Any]
        if !folder.isEmpty { trigger = ["kind": "folder_changed", "path": folder] }
        else if interval > 0 { trigger = ["kind": "interval", "seconds": interval] }
        else { trigger = ["kind": "once", "at": formatter.string(from: date)] }
        let record: [String: Any] = ["id": UUID().uuidString.lowercased(), "name": name, "request": request,
            "conversation_id": UUID().uuidString.lowercased(), "trigger": trigger, "enabled": true,
            "next_run_at": formatter.string(from: date), "last_condition": false]
        if let data = try? JSONSerialization.data(withJSONObject: record) {
            var command = Sage_Ipc_V2_WorkflowCommand()
            command.operation = "save_schedule"; command.json = String(decoding: data, as: UTF8.self)
            command.backgroundExpiresAtUnixMs = Int64(Date().addingTimeInterval(30 * 86400).timeIntervalSince1970 * 1000)
            command.maximumRuns = 100
            if !folder.isEmpty {
                var scope = Sage_Ipc_V2_ResourceScope(); scope.root = folder; scope.effects = [.read]
                command.backgroundResources = [scope]
            }
            Task { do { try await client.workflow(command) } catch { errorMessage = error.localizedDescription } }
        }
    }

    func resume(taskID: String) {
        Task { do { try await client.control(taskID: taskID, operation: .resume); refreshState() } catch { errorMessage = error.localizedDescription } }
    }

    func toggleVoiceInput() {
        voiceNotice = nil
        switch voiceState {
        case .listening:
            voiceInput.finishCurrentCommand()
        case .idle, .wakeListening, .processing:
            Task { await voiceInput.startFromMicrophoneButton() }
        }
    }

    func cancelVoiceInput() {
        voiceInput.cancelCurrentCommand()
    }

    func setWakeWordEnabled(_ enabled: Bool) {
        wakeWordEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "wakeWordEnabled")
        voiceInput.configureWakeWord(enabled: enabled, phrase: wakePhrase)
    }

    func setWakePhrase(_ phrase: String) {
        let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        wakePhrase = normalized.isEmpty ? "Hey Sage" : normalized
        UserDefaults.standard.set(wakePhrase, forKey: "wakePhrase")
        voiceInput.configureWakeWord(enabled: wakeWordEnabled, phrase: wakePhrase)
    }

    func saveProviderSettings(
        provider: String,
        model: String,
        endpoint: String,
        apiKey: String,
        removeSavedKey: Bool
    ) {
        providerSaveMessage = nil
        providerSaving = true
        Task {
            do {
                let changesCredential = removeSavedKey
                    || !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                let authenticated: Bool
                if changesCredential {
                    authenticated = try await NativeAuthentication.authenticate(
                        reason: "Save this model provider credential in macOS Keychain."
                    )
                    guard authenticated else {
                        providerSaving = false
                        return
                    }
                } else {
                    authenticated = false
                }
                try await client.saveProviderSettings(
                    provider: provider,
                    model: model,
                    endpoint: endpoint,
                    apiKey: apiKey,
                    removeSavedKey: removeSavedKey,
                    nativeAuthenticationSatisfied: authenticated
                )
            } catch {
                providerSaving = false
                errorMessage = error.localizedDescription
            }
        }
    }

    func testProviderConnection(
        provider: String,
        model: String,
        endpoint: String,
        apiKey: String
    ) {
        providerTestMessage = nil
        providerTesting = true
        Task {
            do {
                try await client.testProviderConnection(
                    provider: provider,
                    model: model,
                    endpoint: endpoint,
                    apiKey: apiKey
                )
            } catch {
                providerTesting = false
                providerTestMessage = error.localizedDescription
            }
        }
    }

    var microphonePermissionLabel: String {
        switch voiceInput.microphoneAuthorization {
        case .authorized: "Allowed"
        case .notDetermined: "Asked when you use the mic"
        case .denied: "Off in System Settings"
        case .restricted: "Restricted"
        @unknown default: "Unavailable"
        }
    }

    var speechPermissionLabel: String {
        switch voiceInput.speechAuthorization {
        case .authorized: "Allowed"
        case .notDetermined: "Asked after microphone access"
        case .denied: "Off in System Settings"
        case .restricted: "Restricted"
        @unknown default: "Unavailable"
        }
    }

    private func connectWithRetry() async throws {
        var finalError: Error?
        for _ in 0..<30 {
            do {
                try await client.connect()
                return
            } catch {
                finalError = error
                try await Task.sleep(for: .milliseconds(150))
            }
        }
        throw finalError ?? SageClientError.connectionFailed("SAGE Core did not open its socket")
    }

    private func consume(_ event: Sage_Ipc_V2_CoreEvent) {
        switch event.event {
        case .stateSnapshot(let snapshot):
            storageLocked = snapshot.storageLocked
            pendingApproval = snapshot.pendingApprovals.first
            tasks = snapshot.tasks
            if snapshot.hasKnowledge { applyKnowledge(snapshot.knowledge.json) }
            if let conversation = selectedConversationID, let latest = tasks.first(where: { $0.conversationID == conversation }) { selectedTaskID = latest.taskID }
            providerSettings = snapshot.providerSettings.first
            if providerSaving {
                providerSaving = false
                providerSaveMessage = "Saved"
            }
            if selectedTaskID == nil, !draftActive {
                selectedTaskID = visibleTasks.first?.taskID
                selectedConversationID = visibleTasks.first?.conversationID
                knowledge("list")
                if let selectedTaskID {
                    timeline = timelinesByTaskID[selectedTaskID] ?? []
                }
            }
        case .taskUpdate(let task):
            if let index = tasks.firstIndex(where: { $0.taskID == task.taskID }) {
                tasks[index] = task
            } else {
                tasks.insert(task, at: 0)
            }
            if task.taskID == selectedTaskID {
                timeline = timelinesByTaskID[task.taskID] ?? timeline
            }
            if selectedTaskID == nil, !draftActive {
                selectedTaskID = task.taskID
            }
        case .agentEvent(let agentEvent):
            guard agentEvent.kind != "notification" else { return }
            if !agentEvent.taskID.isEmpty {
                timelinesByTaskID[agentEvent.taskID, default: []].insert(agentEvent, at: 0)
                timelinesByTaskID[agentEvent.taskID] = Array(
                    timelinesByTaskID[agentEvent.taskID, default: []].prefix(200)
                )
                if selectedTaskID == nil, !draftActive {
                    selectedTaskID = agentEvent.taskID
                }
                if selectedTaskID == agentEvent.taskID {
                    timeline = timelinesByTaskID[agentEvent.taskID] ?? []
                }
                isSubmitting = false
            } else {
                timeline.insert(agentEvent, at: 0)
                timeline = Array(timeline.prefix(200))
            }
            Task { try? await client.requestState(includeCompleted: true) }
        case .approvalRequest(let approval):
            pendingApproval = approval
        case .questionRequest(let question):
            pendingQuestion = question
        case .error(let error):
            isSubmitting = false
            providerSaving = false
            providerTesting = false
            errorMessage = error.message
            Task { try? await client.requestState(includeCompleted: true) }
        case .providerConnectionResult(let result):
            providerTesting = false
            providerTestMessage = result.success ? result.message : "Test failed: \(result.message)"
        case .knowledgeState(let state): applyKnowledge(state.json)
        case .workflowState(let state):
            if let data = try? decodeKnowledge(WorkflowData.self, state.json) { skills = data.skills; workflows = data.workflows; schedules = data.schedules }
        case .notification:
            isSubmitting = false
            refreshState()
            knowledge("list")
        case .modelResponseDelta(let delta):
            if let index = tasks.firstIndex(where: { $0.taskID == delta.taskID }) {
                if !delta.finished { tasks[index].finalOutcome = delta.text }
            }
            if delta.finished { refreshState(); knowledge("list") }
        case .permissionRequest, nil:
            break
        }
    }

    private func configureVoiceInput() {
        voiceInput.onStateChange = { [weak self] state in
            guard let self else { return }
            voiceState = state
            switch state {
            case .idle, .wakeListening:
                voiceTranscript = ""
                voiceOverlay.hide()
            case .listening:
                voiceOverlay.show(phase: .listening, text: voiceTranscript)
            case .processing:
                voiceOverlay.show(phase: .processing, text: voiceTranscript)
            }
        }
        voiceInput.onTranscript = { [weak self] transcript in
            guard let self else { return }
            voiceTranscript = transcript
            voiceOverlay.show(phase: .listening, text: transcript)
        }
        voiceInput.onCommand = { [weak self] command, _ in
            guard let self else { return }
            voiceTranscript = command
            composerText = command
            voiceOverlay.show(phase: .processing, text: command)
            submit(source: .voice)
        }
        voiceInput.onError = { [weak self] message in
            guard let self else { return }
            voiceNotice = message
            voiceOverlay.show(phase: .error, text: message)
        }
    }

    private func voiceInteractionExecuting(_ transcript: String) {
        voiceOverlay.show(phase: .executing, text: transcript)
    }

    private func persistTaskMetadata() {
        guard let data = try? JSONEncoder().encode(taskMetadata) else { return }
        UserDefaults.standard.set(data, forKey: Self.taskMetadataKey)
    }

    private func isFinished(_ status: Sage_Ipc_V2_TaskStatus) -> Bool {
        [.succeeded, .failed, .cancelled, .interrupted].contains(status)
    }

    private var selectedTaskIsActive: Bool {
        guard let selectedTaskID,
              let task = tasks.first(where: { $0.taskID == selectedTaskID }) else {
            return false
        }
        return !isFinished(task.status)
    }
}
