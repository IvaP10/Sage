import AppKit
import AVFoundation
import CoreFoundation
import Foundation
import Observation
import Speech

struct LearningCandidate: Identifiable, Hashable {
    let id: String
    let kind: String
    let role: String
    let label: String
    let systemID: String
    let systemFingerprint: String
    let systemLabel: String

    var decisionID: String { "learning:\(systemFingerprint):\(id)" }
}

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
    enum UserDecision: Identifiable {
        case approval(Sage_Ipc_V2_ApprovalRequest)
        case question(Sage_Ipc_V2_QuestionRequest)
        case learning(LearningCandidate)
        case controllerDraft(ControllerDraftRecord)
        var id: String {
            switch self {
            case .approval(let value): value.approvalID
            case .question(let value): value.questionID
            case .learning(let value): value.decisionID
            case .controllerDraft(let value): "controller:\(value.id)"
            }
        }
        var taskID: String {
            switch self {
            case .approval(let value): value.taskID
            case .question(let value): value.taskID
            case .learning, .controllerDraft: ""
            }
        }
        var expiresAt: Int64 {
            switch self {
            case .approval(let value): value.expiresAtUnixMs
            case .question(let value): value.expiresAtUnixMs
            case .learning, .controllerDraft: Int64.max
            }
        }
    }
    var pendingDecision: UserDecision?
    private(set) var decisionCount = 0
    private(set) var resolvingDecision = false
    private var decisionInbox: [String: UserDecision] = [:]
    private var deferredDecisions: Set<String> = []
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
    var routineLearningEnabled = false
    var routines: [RoutineRecord] = []
    var routineFamilies: [RoutineFamilyRecord] = []
    var worldModelSnapshotJSON = ""
    var worldModelStatus = ""
    var upnpRendererCandidates: [RendererCandidateRecord] = []
    var upnpRendererTransportStates: [String: String] = [:]
    var upnpRendererProtocolSummaries: [String: String] = [:]
    private(set) var worldModelBusy = false
    var goalBuilderPresented = false
    var fileStreamCopyPresented = false
    private(set) var fileStreamCopyStatus = ""
    private(set) var fileStreamCopyNeedsRetry = false
    private(set) var goalSystems: [GoalSystemOption] = []
    var selectedGoalSystemID = ""
    private(set) var goalCapabilities: [GoalCapabilityOption] = []
    var selectedGoalCapabilityID = ""
    var goalStatus = ""
    private(set) var goalPlanPreview: GoalPlanPreview?
    private(set) var goalSubmissionNeedsRetry = false
    var controllerDraftTaskID: String?
    var controllerDraftStatus = ""
    var controllerDrafts: [ControllerDraftRecord] = []
    private enum ControllerRequest {
        case compile(taskID: String)
        case inspect(id: String)
        case review(id: String)
        case run(id: String)
    }
    private var controllerRequest: ControllerRequest?
    private enum GoalRequest {
        case list
        case system
        case preview
        case run
        case fileStreamCopy
    }
    @ObservationIgnored private var goalRequest: GoalRequest?
    @ObservationIgnored private var preparedGoalSynthesisPayload: Data?
    @ObservationIgnored private var goalRunSubmissionID: String?
    @ObservationIgnored private var goalRunSubmissionPayload: Data?
    @ObservationIgnored private var fileStreamCopySubmissionID: String?
    @ObservationIgnored private var fileStreamCopySubmissionPayload: Data?
    private(set) var learningCandidates: [LearningCandidate] = []
    private(set) var learningSessionID: String?
    private(set) var learningApprovedCandidate: LearningCandidate?
    private(set) var learningExpiresAt: Date?
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
    private(set) var isSpeaking = false
    var wakeWordEnabled = UserDefaults.standard.object(forKey: "wakeWordEnabled") as? Bool ?? false
    var wakePhrase = UserDefaults.standard.string(forKey: "wakePhrase") ?? "Hey Sage"
    private(set) var isSubmitting = false
    private(set) var draftActive = false
    private(set) var streamedResponses: [String: String] = [:]
    private(set) var stoppingTaskIDs: Set<String> = []
    var refreshNotice: String?
    private(set) var intentPreview: Sage_Ipc_V2_IntentPreview?
    private let intentStreamID = UUID().uuidString.lowercased()
    private var intentRevision: UInt64 = 0
    private var voiceStreamActive = false
    private var voiceStreamPrefixSubmitted = false
    private var streamedVoicePrefixText: String?
    private var activeStreamedVoiceTaskID: String?
    private var streamedPrefixRequestIDs: Set<String> = []
    private var streamedFinalRequestIDs: Set<String> = []
    @ObservationIgnored private var intentPreparation: Task<Void, Never>?

    private let supervisor = CoreSupervisor()
    private let client = SageCoreClient()
    private var reconnecting = false
    private var unconfirmedRequest: String?
    private let platformAdapter = PlatformAdapter()
    private let voiceOverlay = OverlayWindowController()
    private let voiceInput = VoiceInputController()
    private let speechOutput = SpeechOutputController()
    private var voiceTurnGeneration = 0
    private var voiceRequestGenerations: [String: Int] = [:]
    private var voiceTaskGenerations: [String: Int] = [:]
    private var voiceSentenceBuffers: [String: ResponseSentenceBuffer] = [:]
    private var activeSpeechTaskID: String?
    private var taskMetadata: [String: TaskMetadata] = [:]
    private var timelinesByTaskID: [String: [Sage_Ipc_V2_AgentEvent]] = [:]
    private var lastSubmittedRequest = ""
    private var lastSubmittedAt: Date?
    private var started = false
    @ObservationIgnored private var snapshotRefresh: PresentationRefreshCoordinator?
    @ObservationIgnored private var historyRefresh: PresentationRefreshCoordinator?
    @ObservationIgnored private var responsePresentation: Task<Void, Never>?
    @ObservationIgnored private var pendingResponseText: [String: String] = [:]

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
                let lhsPinned = isPinned(lhs.element.taskID)
                let rhsPinned = isPinned(rhs.element.taskID)
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
        let conversationID = tasks.first(where: { $0.taskID == taskID })?.conversationID ?? ""
        selectedConversationID = conversationID.isEmpty ? nil : conversationID
        messages = []
        refreshConversationMessages()
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
            selectedConversationID = nil
            messages = []
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
        client.onEvent = { [weak self] event in
            Task { @MainActor in self?.consume(event) }
        }
        client.onDisconnect = { [weak self] message in
            Task { @MainActor in
                guard let self, self.started else { return }
                if self.composerText.isEmpty {
                    self.composerText = self.unconfirmedRequest
                        ?? (self.voiceStreamPrefixSubmitted
                            ? (self.voiceTranscript.isEmpty
                                ? (self.streamedVoicePrefixText ?? "")
                                : self.voiceTranscript)
                            : "")
                }
                self.unconfirmedRequest = nil
                self.isSubmitting = false
                self.draftActive = !self.composerText.isEmpty
                self.resetStreamedVoiceState()
                self.connectionState = .failed(message)
                self.snapshotRefresh?.reset()
                self.historyRefresh?.reset()
                self.resolvingDecision = false
                if let goalRequest = self.goalRequest {
                    self.worldModelBusy = false
                    self.goalRequestFailed(goalRequest, message: "The Core connection closed before Sage confirmed this goal operation.")
                }
                if self.learningSessionID != nil {
                    self.learningSessionID = nil
                    self.learningApprovedCandidate = nil
                    self.learningExpiresAt = nil
                    self.worldModelBusy = false
                    self.worldModelStatus = "The native connection closed; learning approval ended and any unsettled probe needs review."
                    self.showNextDecision()
                }
                self.reconnect()
            }
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
            requestSnapshot(immediate: true)
            voiceInput.configureWakeWord(enabled: wakeWordEnabled, phrase: wakePhrase)
        } catch {
            connectionState = .failed(error.localizedDescription)
            errorMessage = error.localizedDescription
        }
    }

    func stop() {
        started = false
        snapshotRefresh?.reset()
        historyRefresh?.reset()
        responsePresentation?.cancel()
        responsePresentation = nil
        pendingResponseText.removeAll()
        speechOutput.stopImmediately()
        clearSpokenReplyState()
        resetStreamedVoiceState()
        voiceInput.stop()
        voiceOverlay.hide()
        client.disconnect()
        supervisor.detach()
    }

    private func resetStreamedVoiceState() {
        voiceStreamActive = false
        voiceStreamPrefixSubmitted = false
        streamedVoicePrefixText = nil
        activeStreamedVoiceTaskID = nil
        streamedPrefixRequestIDs.removeAll()
        streamedFinalRequestIDs.removeAll()
    }

    func submit(
        source: Sage_Ipc_V2_InputSource = .typed,
        requestOverride: String? = nil,
        streamedPrefix: Bool = false,
        finalizeStream: Bool = false
    ) {
        let request = (requestOverride ?? composerText).trimmingCharacters(in: .whitespacesAndNewlines)
        let earlyPrefix = streamedPrefix
        guard connectionState == .connected, !request.isEmpty,
              earlyPrefix || finalizeStream || !isSubmitting else { return }
        if earlyPrefix {
            guard voiceStreamActive, !voiceStreamPrefixSubmitted,
                  !selectedTaskIsActive,
                  composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            voiceStreamPrefixSubmitted = true
            streamedVoicePrefixText = request
        }
        let supersedesTaskID = (earlyPrefix || finalizeStream)
            ? ""
            : (selectedTaskIsActive ? (selectedTaskID ?? "") : "")
        if !earlyPrefix,
           request == lastSubmittedRequest,
           let lastSubmittedAt,
           Date().timeIntervalSince(lastSubmittedAt) < 0.6 {
            return
        }
        if !earlyPrefix {
            lastSubmittedRequest = request
            lastSubmittedAt = Date()
        }
        let voiceGeneration: Int?
        let submissionRequestID: String?
        if source == .voice {
            if !finalizeStream && !earlyPrefix { invalidateVoiceRepliesForNewInput() }
            voiceGeneration = voiceTurnGeneration
            submissionRequestID = UUID().uuidString.lowercased()
            if let submissionRequestID, !finalizeStream {
                voiceRequestGenerations[submissionRequestID] = voiceGeneration
            }
            if let submissionRequestID, earlyPrefix {
                streamedPrefixRequestIDs.insert(submissionRequestID)
            }
            if let submissionRequestID, finalizeStream {
                streamedFinalRequestIDs.insert(submissionRequestID)
            }
        } else {
            voiceGeneration = nil
            submissionRequestID = nil
        }
        if !earlyPrefix {
            isSubmitting = true
            unconfirmedRequest = request
            draftActive = false
            composerText = ""
        }
        if selectedConversationID == nil { selectedConversationID = UUID().uuidString.lowercased() }
        let submissionConversationID = selectedConversationID ?? ""
        let submissionFolders = taskFolders
        Task { [self] in
            do {
                try await client.submitTask(
                    request,
                    source: source,
                    conversationID: submissionConversationID,
                    folders: submissionFolders,
                    supersedesTaskID: supersedesTaskID,
                    voiceStreamID: (earlyPrefix || finalizeStream) ? intentStreamID : "",
                    streamedPrefix: earlyPrefix,
                    finalizeStream: finalizeStream,
                    requestID: submissionRequestID,
                    onRequestIDStaged: { [weak self] stagedRequestID in
                        guard let self else { return }
                        if let submissionRequestID, submissionRequestID != stagedRequestID {
                            self.voiceRequestGenerations.removeValue(forKey: submissionRequestID)
                            self.streamedPrefixRequestIDs.remove(submissionRequestID)
                            self.streamedFinalRequestIDs.remove(submissionRequestID)
                        }
                        if let voiceGeneration, !finalizeStream {
                            self.voiceRequestGenerations[stagedRequestID] = voiceGeneration
                        }
                        if earlyPrefix { self.streamedPrefixRequestIDs.insert(stagedRequestID) }
                        if finalizeStream { self.streamedFinalRequestIDs.insert(stagedRequestID) }
                    }
                )
                if !earlyPrefix { taskFolders = [] }
                if source == .voice && !earlyPrefix {
                    voiceInteractionExecuting(request)
                }
            } catch {
                if let submissionRequestID {
                    voiceRequestGenerations.removeValue(forKey: submissionRequestID)
                    streamedPrefixRequestIDs.remove(submissionRequestID)
                    streamedFinalRequestIDs.remove(submissionRequestID)
                }
                if earlyPrefix {
                    voiceStreamPrefixSubmitted = false
                    voiceNotice = "Sage could not start the prepared step. Your speech remains a draft."
                } else {
                    isSubmitting = false
                    draftActive = true
                    if composerText.isEmpty { composerText = request }
                }
                errorMessage = error.localizedDescription
            }
        }
    }

    func prepareIntent(_ text: String, voice: Bool = false) {
        intentPreparation?.cancel()
        intentRevision += 1
        intentPreview = nil
        guard connectionState == .connected else { return }
        // Empty input retires the previous native lookup as well as its card.
        // Oversized drafts also cancel preparation without transmitting them.
        let draft = text.utf8.count <= 4096 ? text : ""
        let revision = intentRevision
        let taskID = selectedTaskIsActive ? (selectedTaskID ?? "") : ""
        let folders = taskFolders
        intentPreparation = Task { [weak self] in
            guard let self else { return }
            do {
                if !voice && !draft.isEmpty { try await Task.sleep(for: .milliseconds(120)) }
                try Task.checkCancellation()
                try await client.updateIntent(streamID: intentStreamID, revision: revision, text: draft,
                    activeTaskID: taskID, allowInterrupt: voice, voiceInput: voice, folders: folders)
            } catch is CancellationError { }
            catch {
                // A preview failure must not turn a partially typed input into
                // a submitted task or clear the user's draft.
                if voice && revision == intentRevision {
                    voiceNotice = "Could not confirm voice control. Use the Stop button to stop the task."
                }
            }
        }
    }

    private func cancelStreamedVoicePrefixIfNeeded() {
        guard voiceStreamPrefixSubmitted else { return }
        voiceStreamActive = false
        prepareIntent("Stop", voice: true)
    }

    private func remainingVoiceRequest(after prefix: String, in request: String) -> String? {
        guard let prefixRange = request.range(
            of: prefix,
            options: [.anchored, .caseInsensitive]
        ) else { return nil }
        var remainder = String(request[prefixRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let sequence = remainder.range(
            of: "and then",
            options: [.anchored, .caseInsensitive]
        ) else { return nil }
        let afterSequence = remainder[sequence.upperBound...]
        guard afterSequence.first.map({ $0.isWhitespace || $0.isPunctuation }) ?? true else {
            return nil
        }
        remainder = String(afterSequence)
            .trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
        return remainder.isEmpty ? nil : remainder
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
        guard !resolvingDecision else { return }
        resolvingDecision = true
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
                if approval.requiresNativeAuthentication && !authenticated {
                    resolvingDecision = false
                    return
                }
                try await client.resolveApproval(
                    approval,
                    approve: true,
                    nativeAuthenticationSatisfied: authenticated
                )
            } catch {
                resolvingDecision = false
                errorMessage = error.localizedDescription
            }
        }
    }

    func deny(_ approval: Sage_Ipc_V2_ApprovalRequest) {
        guard !resolvingDecision else { return }
        resolvingDecision = true
        Task {
            do {
                try await client.resolveApproval(
                    approval,
                    approve: false,
                    nativeAuthenticationSatisfied: false
                )
            } catch {
                resolvingDecision = false
                errorMessage = error.localizedDescription
            }
        }
    }

    func answer(_ answer: String, question: Sage_Ipc_V2_QuestionRequest) {
        guard !resolvingDecision else { return }
        resolvingDecision = true
        Task {
            do {
                try await client.answer(question, text: answer)
            } catch {
                resolvingDecision = false
                errorMessage = error.localizedDescription
            }
        }
    }

    func cancel(taskID: String) {
        guard stoppingTaskIDs.insert(taskID).inserted else { return }
        stopSpokenReplyIfNeeded(for: taskID)
        Task {
            do {
                try await client.control(taskID: controlScope(for: taskID), operation: .cancel)
                requestSnapshot(immediate: true)
            } catch {
                stoppingTaskIDs.remove(taskID)
                errorMessage = error.localizedDescription
            }
        }
    }

    func deferDecision() {
        if let decision = pendingDecision { deferredDecisions.insert(decision.id) }
        pendingDecision = nil
        showNextDecision()
    }

    private func controlScope(for taskID: String) -> String {
        guard let task = tasks.first(where: { $0.taskID == taskID }), !task.controlScopeID.isEmpty else { return taskID }
        return task.controlScopeID
    }

    func reviewDecisions() {
        deferredDecisions.removeAll()
        showNextDecision()
    }

    func reviewDecision(taskID: String) {
        deferredDecisions = deferredDecisions.filter { decisionInbox[$0]?.taskID != taskID }
        pendingDecision = decisionInbox.values
            .filter { $0.taskID == taskID && $0.expiresAt > Int64(Date().timeIntervalSince1970 * 1_000) }
            .sorted { $0.expiresAt < $1.expiresAt }.first
        showNextDecision()
    }

    func hasDecision(taskID: String) -> Bool {
        decisionInbox.values.contains { $0.taskID == taskID && $0.expiresAt > Int64(Date().timeIntervalSince1970 * 1_000) }
    }

    func prepareFollowUp(_ task: Sage_Ipc_V2_TaskUpdate) {
        composerText = task.request
        focusComposer()
    }

    func openContinuation(_ task: Sage_Ipc_V2_TaskUpdate) {
        guard !task.continuedTaskID.isEmpty else { return }
        if tasks.contains(where: { $0.taskID == task.continuedTaskID }) {
            selectTask(task.continuedTaskID)
        } else {
            selectedTaskID = task.continuedTaskID
            timeline = timelinesByTaskID[task.continuedTaskID] ?? []
            requestSnapshot(immediate: true)
        }
    }

    private func showNextDecision() {
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        decisionInbox = decisionInbox.filter { $0.value.expiresAt > now }
        decisionCount = decisionInbox.count
        if let current = pendingDecision, decisionInbox[current.id] == nil {
            pendingDecision = nil
            resolvingDecision = false
        }
        if pendingDecision == nil {
            pendingDecision = decisionInbox.values.filter { !deferredDecisions.contains($0.id) }
                .sorted { $0.expiresAt < $1.expiresAt }.first
        }
    }

    private func removeDecision(_ id: String) {
        decisionInbox.removeValue(forKey: id)
        deferredDecisions.remove(id)
        showNextDecision()
    }

    func undo(taskID: String) {
        guard let actionID = tasks.first(where: { $0.taskID == taskID })?.undoActionID,
              !actionID.isEmpty else {
            errorMessage = "Refresh or update Sage before Undo; the action identity is missing."
            return
        }
        Task {
            do {
                try await client.undo(taskID: taskID, actionID: actionID)
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
        requestSnapshot(immediate: true)
        refreshConversationMessages()
    }

    private func requestSnapshot(immediate: Bool = false) {
        guard connectionState == .connected else { return }
        if snapshotRefresh == nil {
            snapshotRefresh = PresentationRefreshCoordinator(send: { [weak self] in
                guard let self else { return }
                try await client.requestState(includeCompleted: true)
            }, failed: { [weak self] error in
                self?.refreshNotice = error.localizedDescription
            })
        }
        snapshotRefresh?.request(immediate: immediate)
    }

    private func refreshConversationMessages() {
        guard connectionState == .connected, selectedConversationID != nil else { return }
        if historyRefresh == nil {
            historyRefresh = PresentationRefreshCoordinator(send: { [weak self] in
                guard let self, let conversationID = selectedConversationID else { return }
                var request = Sage_Ipc_V2_KnowledgeCommand()
                request.operation = "list"
                request.conversationID = conversationID
                try await client.knowledge(request)
            }, failed: { [weak self] error in
                self?.refreshNotice = error.localizedDescription
            })
        }
        historyRefresh?.request()
    }

    private func applyKnowledge(_ json: String) {
        guard let data = try? decodeKnowledge(KnowledgeData.self, json) else { return }
        conversations = data.conversations; memories = data.memories; memoryEnabled = data.memoryEnabled
        let incoming = data.messages.filter { $0.conversationId == selectedConversationID }
        if !incoming.isEmpty { messages = incoming }
    }

    func knowledge(_ operation: String, id: String = "", content: String = "", enabled: Bool = true) {
        if operation == "list", selectedConversationID != nil { refreshConversationMessages(); return }
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
                if archived && !isFinished(task.status) { try await client.control(taskID: controlScope(for: task.taskID), operation: .cancel) }
                try await client.knowledge(request)
                requestSnapshot()
            } catch { errorMessage = error.localizedDescription }
        }
    }

    func workflow(_ operation: String, id: String = "", name: String = "", json: String = "") {
        var request = Sage_Ipc_V2_WorkflowCommand()
        request.operation = operation; request.id = id; request.name = name; request.json = json
        request.conversationID = selectedConversationID ?? ""
        Task { do { try await client.workflow(request) } catch { errorMessage = error.localizedDescription } }
    }

    func openGoalBuilder() {
        guard !worldModelBusy, learningSessionID == nil, pendingDecision == nil else { return }
        goalBuilderPresented = true
        if goalRunSubmissionPayload != nil {
            goalStatus = "The previous submission has no confirmed response. Retry it with its original submission identity."
            return
        }
        refreshGoalSystems()
    }

    func refreshGoalSystems() {
        guard !worldModelBusy, goalRunSubmissionPayload == nil else { return }
        goalCapabilities = []
        selectedGoalCapabilityID = ""
        invalidateGoalPreview()
        goalRequest = .list
        worldModelBusy = true
        goalStatus = "Refreshing discovered application systems…"
        worldModel("list", id: UUID().uuidString.lowercased())
    }

    func goalSystemSelectionChanged() {
        guard goalRunSubmissionPayload == nil else { return }
        goalCapabilities = []
        selectedGoalCapabilityID = ""
        invalidateGoalPreview()
        if !selectedGoalSystemID.isEmpty {
            goalStatus = "Load the selected application's current verified controls."
        }
    }

    func loadGoalCapabilities() {
        guard !worldModelBusy, goalRunSubmissionPayload == nil,
              goalSystems.contains(where: { $0.id == selectedGoalSystemID }) else { return }
        goalCapabilities = []
        selectedGoalCapabilityID = ""
        invalidateGoalPreview()
        goalRequest = .system
        worldModelBusy = true
        goalStatus = "Checking current evidence for the selected application…"
        worldModel("system", id: selectedGoalSystemID)
    }

    func invalidateGoalPreview() {
        guard goalRunSubmissionPayload == nil else { return }
        goalPlanPreview = nil
        preparedGoalSynthesisPayload = nil
    }

    func previewGoal(capability: GoalCapabilityOption, value: GoalLiteralValue) {
        guard !worldModelBusy, goalRunSubmissionPayload == nil,
              capability.systemID == selectedGoalSystemID,
              let system = goalSystems.first(where: { $0.id == capability.systemID }),
              system.fingerprint == capability.systemFingerprint else { return }
        let valueType: String
        let encodedValue: Any
        switch value {
        case .number(let number):
            guard capability.input.valueType == "number", number.isFinite else {
                goalStatus = "Enter a finite number for this control."
                return
            }
            valueType = "number"
            encodedValue = number
        case .boolean(let boolean):
            guard capability.input.valueType == "boolean" else {
                goalStatus = "Choose a state that matches this control's typed input."
                return
            }
            valueType = "boolean"
            encodedValue = boolean
        }
        let seed: [String: Any] = [
            "capability_id": capability.id,
            "input_name": capability.input.name,
            "binding": [
                "source": "literal",
                "value": ["type": valueType, "value": encodedValue],
                "port": capability.input.wireObject,
            ],
        ]
        let payload: [String: Any] = [
            "system_id": capability.systemID,
            "goal_capability_id": capability.id,
            "goal_output": capability.output.name,
            "seeds": [seed],
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else {
            goalStatus = "Sage could not encode this bounded typed goal."
            return
        }
        preparedGoalSynthesisPayload = data
        goalPlanPreview = nil
        goalRequest = .preview
        worldModelBusy = true
        goalStatus = "Synthesizing a proposal from current restored evidence…"
        worldModel("synthesize_goal", id: UUID().uuidString.lowercased(), json: json)
    }

    func runPreparedGoal(taskDescription: String) {
        guard !worldModelBusy, goalSubmissionNeedsRetry == false,
              goalPlanPreview != nil,
              let synthesisData = preparedGoalSynthesisPayload,
              var payload = try? JSONSerialization.jsonObject(with: synthesisData) as? [String: Any] else { return }
        let request = taskDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty, request.utf8.count <= 512 else {
            goalStatus = "Add a task description under 512 bytes before starting."
            return
        }
        payload["request"] = request
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
            goalStatus = "Sage could not encode this reviewed goal submission."
            return
        }
        goalRunSubmissionID = UUID().uuidString.lowercased()
        goalRunSubmissionPayload = data
        dispatchGoalSubmission()
    }

    func retryGoalSubmission() {
        guard goalSubmissionNeedsRetry, goalRunSubmissionPayload != nil else { return }
        dispatchGoalSubmission()
    }

    private func dispatchGoalSubmission() {
        guard !worldModelBusy, let id = goalRunSubmissionID,
              let data = goalRunSubmissionPayload,
              let json = String(data: data, encoding: .utf8) else { return }
        goalSubmissionNeedsRetry = false
        goalRequest = .run
        worldModelBusy = true
        goalStatus = "Submitting the same declared goal. Sage will recheck the target and ask approval before each action…"
        worldModel("run_goal", id: id, json: json)
    }

    func openFileStreamCopy() {
        guard connectionState == .connected,
              !worldModelBusy,
              learningSessionID == nil,
              pendingDecision == nil else { return }
        fileStreamCopyStatus = "Choose one source file and an exact destination."
        fileStreamCopyPresented = true
    }

    func submitFileStreamCopy(sourcePath: String, destinationPath: String, overwrite: Bool) {
        guard !worldModelBusy,
              fileStreamCopySubmissionPayload == nil,
              !sourcePath.isEmpty,
              !destinationPath.isEmpty else { return }
        let payload: [String: Any] = [
            "source_path": sourcePath,
            "destination_path": destinationPath,
            "overwrite": overwrite,
            "request": "Copy the selected local file to the selected destination",
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]) else {
            fileStreamCopyStatus = "Sage could not encode this bounded file-copy request."
            return
        }
        fileStreamCopySubmissionID = UUID().uuidString.lowercased()
        fileStreamCopySubmissionPayload = data
        fileStreamCopyNeedsRetry = false
        dispatchFileStreamCopy()
    }

    func retryFileStreamCopy() {
        guard fileStreamCopyNeedsRetry, fileStreamCopySubmissionPayload != nil else { return }
        dispatchFileStreamCopy()
    }

    private func dispatchFileStreamCopy() {
        guard !worldModelBusy,
              let id = fileStreamCopySubmissionID,
              let data = fileStreamCopySubmissionPayload,
              let json = String(data: data, encoding: .utf8) else { return }
        fileStreamCopyNeedsRetry = false
        goalRequest = .fileStreamCopy
        worldModelBusy = true
        fileStreamCopyStatus = "Submitting the exact paths. Sage will resolve them again and request approval before reading or writing."
        worldModel("run_file_stream_copy", id: id, json: json)
    }

    func worldModel(_ operation: String, id: String = "", json: String = "") {
        var request = Sage_Ipc_V2_WorldModelCommand()
        request.operation = operation
        request.id = id
        request.json = json
        Task {
            do { try await client.worldModel(request) }
            catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
                if let goalRequest {
                    goalRequestFailed(goalRequest, message: error.localizedDescription)
                }
                if controllerRequest != nil {
                    controllerDraftStatus = error.localizedDescription
                    controllerRequest = nil
                }
                errorMessage = error.localizedDescription
            }
        }
    }

    private func goalRequestFailed(_ request: GoalRequest, message: String) {
        switch request {
        case .run:
            // The command identity and exact payload remain stable so retry is idempotent
            // even when Core committed the task but its receipt was lost.
            goalRequest = nil
            goalSubmissionNeedsRetry = true
            goalStatus = "Sage could not confirm the result. Retry the same submission to safely recover its task: \(message)"
        case .list:
            goalRequest = nil
            goalSystems = []
            goalStatus = "Could not load discovered applications: \(message)"
        case .system:
            goalRequest = nil
            goalCapabilities = []
            selectedGoalCapabilityID = ""
            goalStatus = "Could not verify current application controls: \(message)"
        case .preview:
            goalRequest = nil
            goalPlanPreview = nil
            preparedGoalSynthesisPayload = nil
            goalStatus = "Could not prepare a current procedure preview: \(message)"
        case .fileStreamCopy:
            goalRequest = nil
            fileStreamCopyNeedsRetry = true
            fileStreamCopyStatus = "Sage could not confirm the submission. Retry the same request to recover safely: \(message)"
        }
    }

    private func applyGoalResult(_ result: [String: Any]) {
        guard let request = goalRequest else { return }
        switch request {
        case .list:
            guard let values = result["systems"] as? [Any] else {
                goalRequestFailed(request, message: "Core returned no valid system list.")
                return
            }
            let systems = values.compactMap(GoalSystemOption.init)
                .filter { $0.kind == "application" }
                .sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
            goalSystems = systems
            if !systems.contains(where: { $0.id == selectedGoalSystemID }) {
                selectedGoalSystemID = systems.first?.id ?? ""
            }
            goalCapabilities = []
            selectedGoalCapabilityID = ""
            goalPlanPreview = nil
            preparedGoalSynthesisPayload = nil
            goalRequest = nil
            goalStatus = systems.isEmpty
                ? "No discovered application systems are available yet. Passively inspect an app and review a reversible control first."
                : "Choose an application, then load its current verified controls."
        case .system:
            guard let value = result["system"],
                  let system = GoalSystemOption(value),
                  system.id == selectedGoalSystemID,
                  goalSystems.contains(where: { $0.id == system.id && $0.fingerprint == system.fingerprint }),
                  let values = result["capabilities"] as? [Any] else {
                goalRequestFailed(request, message: "The selected system changed or returned an incomplete response. Refresh it before previewing.")
                return
            }
            let capabilities = values.compactMap { GoalCapabilityOption($0, for: system) }
                .sorted { $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending }
            goalCapabilities = capabilities
            selectedGoalCapabilityID = capabilities.first?.id ?? ""
            goalPlanPreview = nil
            preparedGoalSynthesisPayload = nil
            goalRequest = nil
            goalStatus = capabilities.isEmpty
                ? "No current, reversibly tested application controls meet the bounded goal-builder contract."
                : "Select a tested control and typed value to preview a procedure."
        case .preview:
            guard result["requires_fresh_authority_and_runtime_review"] as? Bool == true,
                  let proposal = result["proposal"] as? [String: Any],
                  let nodes = proposal["nodes"] as? [[String: Any]],
                  !nodes.isEmpty, nodes.count <= 128,
                  let schedule = result["schedule"] as? [String: Any],
                  let waveValues = schedule["waves"] as? [[String]],
                  !waveValues.isEmpty, waveValues.count <= 128,
                  let initialAdvance = result["initial_runtime_advance"] as? [String: Any],
                  let firstWave = initialAdvance["proposal_wave"] as? [[String: Any]],
                  !firstWave.isEmpty,
                  let timeNumber = schedule["estimated_elapsed_micros"] as? NSNumber else {
                goalRequestFailed(request, message: "The proposal did not satisfy Sage's bounded preview contract.")
                return
            }
            let capabilityByID = Dictionary(goalCapabilities.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            let nodeIDs = nodes.compactMap { $0["id"] as? String }
            let firstWaveNodeIDs = firstWave.compactMap { $0["node_id"] as? String }
            guard nodeIDs.count == nodes.count,
                  Set(nodeIDs).count == nodeIDs.count,
                  firstWaveNodeIDs.count == firstWave.count,
                  Set(firstWaveNodeIDs).count == firstWaveNodeIDs.count,
                  waveValues.flatMap({ $0 }).sorted() == nodeIDs.sorted(),
                  Set(firstWaveNodeIDs).isSubset(of: Set(nodeIDs)) else {
                goalRequestFailed(request, message: "The procedure schedule did not cover its exact bounded node set.")
                return
            }
            var capabilityByNodeID: [String: GoalCapabilityOption] = [:]
            for node in nodes {
                guard let id = node["id"] as? String,
                      let kind = node["kind"] as? [String: Any],
                      let capabilityID = kind["capability_id"] as? String,
                      let capability = capabilityByID[capabilityID] else { continue }
                capabilityByNodeID[id] = capability
            }
            let steps = nodes.compactMap { node -> GoalPlanStep? in
                guard let id = node["id"] as? String,
                      let kind = node["kind"] as? [String: Any],
                      kind["kind"] as? String == "capability_call",
                      let capabilityID = kind["capability_id"] as? String,
                      let capability = capabilityByID[capabilityID],
                      capabilityByNodeID[id] != nil,
                      let systemID = kind["system_id"] as? String,
                      systemID == capability.systemID,
                      let fingerprint = kind["system_fingerprint"] as? String,
                      fingerprint == capability.systemFingerprint,
                      let outputs = node["outputs"] as? [String: Any],
                      outputs.count == 1,
                      let outputValue = outputs[capability.output.name],
                      let declaredOutput = GoalPortOption(outputValue),
                      declaredOutput == capability.output,
                      let bindings = kind["input_bindings"] as? [String: Any],
                      bindings.count == 1,
                      let binding = bindings[capability.input.name] as? [String: Any],
                      let source = binding["source"] as? String else { return nil }
                let inputSummary: String
                if source == "literal" {
                    guard let portValue = binding["port"],
                          let port = GoalPortOption(portValue),
                          port == capability.input,
                          let typedValue = binding["value"] as? [String: Any],
                          typedValue["type"] as? String == capability.input.valueType,
                          let rawValue = typedValue["value"] else { return nil }
                    if capability.input.valueType == "number",
                       let number = rawValue as? NSNumber,
                       CFGetTypeID(number) != CFBooleanGetTypeID(),
                       number.doubleValue.isFinite {
                        inputSummary = "\(capability.input.name) = \(number.stringValue)"
                    } else if capability.input.valueType == "boolean",
                              let number = rawValue as? NSNumber,
                              CFGetTypeID(number) == CFBooleanGetTypeID() {
                        inputSummary = "\(capability.input.name) = \(number.boolValue ? "on" : "off")"
                    } else {
                        return nil
                    }
                } else if source == "result" {
                    guard let producer = binding["producer"] as? String,
                          let output = binding["output"] as? String,
                          let producerCapability = capabilityByNodeID[producer],
                          producerCapability.output.name == output,
                          producerCapability.output.valueType == capability.input.valueType,
                          producerCapability.output.privacy == capability.input.privacy,
                          producerCapability.output.maximumBytes == capability.input.maximumBytes,
                          let producerIndex = nodeIDs.firstIndex(of: producer) else { return nil }
                    inputSummary = "\(capability.input.name) from verified output of step \(producerIndex + 1)"
                } else {
                    return nil
                }
                return GoalPlanStep(
                    id: id,
                    capabilityID: capabilityID,
                    capabilityLabel: capability.label,
                    inputSummary: inputSummary,
                    effects: capability.effects,
                    verification: capability.verification,
                    restoration: capability.restoration
                )
            }
            guard steps.count == nodes.count,
                  capabilityByID[selectedGoalCapabilityID] != nil,
                  steps.contains(where: { $0.capabilityID == selectedGoalCapabilityID }) else {
                goalRequestFailed(request, message: "The synthesized procedure contains an unsupported or stale capability.")
                return
            }
            goalPlanPreview = GoalPlanPreview(
                steps: steps,
                waves: waveValues,
                estimatedElapsedMicros: timeNumber.uint64Value
            )
            goalRequest = nil
            goalStatus = "Preview prepared from current evidence. No action has run and no authority has been granted."
        case .run:
            guard let taskID = result["task_id"] as? String,
                  UUID(uuidString: taskID) != nil,
                  result["task_submitted"] as? Bool == true || result["deduplicated"] as? Bool == true else {
                goalRequestFailed(request, message: "Core returned no confirmed task identity.")
                return
            }
            goalRequest = nil
            goalRunSubmissionID = nil
            goalRunSubmissionPayload = nil
            goalSubmissionNeedsRetry = false
            goalStatus = "Started task \(taskID). Sage will request fresh approval before each action and verify every result."
            goalBuilderPresented = false
            requestSnapshot(immediate: true)
        case .fileStreamCopy:
            guard let taskID = result["task_id"] as? String,
                  UUID(uuidString: taskID) != nil,
                  result["task_submitted"] as? Bool == true || result["deduplicated"] as? Bool == true,
                  let maximumBytes = result["maximum_bytes"] as? NSNumber,
                  maximumBytes.intValue == 16 * 1024 * 1024 else {
                goalRequestFailed(request, message: "Core returned no confirmed bounded file-copy task identity.")
                return
            }
            goalRequest = nil
            fileStreamCopySubmissionID = nil
            fileStreamCopySubmissionPayload = nil
            fileStreamCopyNeedsRetry = false
            fileStreamCopyStatus = "Started task \(taskID). Review Sage's exact read and write approvals in the decision inbox."
            requestSnapshot(immediate: true)
        }
    }

    func compileControllerDraft(taskID: String) {
        guard !worldModelBusy, learningSessionID == nil else { return }
        controllerDraftTaskID = taskID
        controllerDraftStatus = "Compiling from this task's saved verification evidence…"
        controllerRequest = .compile(taskID: taskID)
        worldModelBusy = true
        worldModel("compile_controller_draft", id: taskID)
    }

    func inspectControllerDraft(_ draft: ControllerDraftRecord) {
        guard !worldModelBusy, pendingDecision == nil else { return }
        controllerRequest = .inspect(id: draft.id)
        controllerDraftStatus = "Loading the exact saved procedure for review…"
        worldModelBusy = true
        worldModel("get_controller_draft", id: draft.id)
    }

    func reviewControllerDraft(_ draft: ControllerDraftRecord) {
        guard !worldModelBusy,
              case .controllerDraft(let current)? = pendingDecision,
              current.id == draft.id else { return }
        let payload: [String: Any] = ["expected_revision": draft.revision]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            controllerDraftStatus = "Sage could not prepare the controller review request."
            return
        }
        controllerRequest = .review(id: draft.id)
        controllerDraftStatus = "Passively checking the current app and rebinding every control…"
        worldModelBusy = true
        worldModel("review_controller_draft", id: draft.id, json: String(decoding: data, as: UTF8.self))
    }

    func runReviewedController(_ draft: ControllerDraftRecord) {
        guard !worldModelBusy,
              draft.status == "reviewed",
              case .controllerDraft(let current)? = pendingDecision,
              current.id == draft.id,
              current.revision == draft.revision else { return }
        let payload: [String: Any] = [
            "controller_id": draft.id,
            "expected_revision": draft.revision,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            controllerDraftStatus = "Sage could not prepare the controller run request."
            return
        }
        deferDecision()
        controllerRequest = .run(id: draft.id)
        controllerDraftStatus = "Checking the exact foreground app. Every action will still require fresh approval."
        worldModelBusy = true
        worldModel("run_controller", id: UUID().uuidString.lowercased(), json: String(decoding: data, as: UTF8.self))
    }

    private func decodeControllerDraft(_ value: Any?) -> ControllerDraftRecord? {
        guard let value,
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return try? decodeKnowledge(ControllerDraftRecord.self, json)
    }

    private func decodeControllerDrafts(_ value: Any?) -> [ControllerDraftRecord]? {
        guard let value,
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return try? decodeKnowledge([ControllerDraftRecord].self, json)
    }

    private func decodeRendererCandidates(_ value: Any?) -> [RendererCandidateRecord]? {
        guard let value,
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return try? decodeKnowledge([RendererCandidateRecord].self, json)
    }

    private func decodeRendererTransportObservation(
        _ value: Any?
    ) -> RendererTransportObservationRecord? {
        guard let value,
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return try? decodeKnowledge(RendererTransportObservationRecord.self, json)
    }

    private func decodeRendererProtocolInfoObservation(
        _ value: Any?
    ) -> RendererProtocolInfoObservationRecord? {
        guard let value,
              let data = try? JSONSerialization.data(withJSONObject: value),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return try? decodeKnowledge(RendererProtocolInfoObservationRecord.self, json)
    }

    private func upsertControllerDraft(_ record: ControllerDraftRecord) {
        controllerDrafts.removeAll { $0.id == record.id }
        controllerDrafts.insert(record, at: 0)
        controllerDrafts = Array(controllerDrafts.prefix(8))
    }

    func approveLearningCandidate(_ candidate: LearningCandidate) {
        guard !worldModelBusy, learningSessionID == nil else { return }
        let payload: [String: Any] = [
            "system_id": candidate.systemID,
            "system_fingerprint": candidate.systemFingerprint,
            "permitted_probes": [candidate.id: candidate.kind],
            "expires_in_seconds": 600,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
            worldModelStatus = "Sage could not prepare this exact-control approval."
            return
        }
        learningApprovedCandidate = candidate
        worldModelBusy = true
        worldModelStatus = "Requesting approval for one reversible test of \(candidate.label)…"
        worldModel("approve_learning_session", json: String(decoding: data, as: UTF8.self))
    }

    func runApprovedLearningProbe() {
        guard !worldModelBusy,
              let sessionID = learningSessionID,
              let candidate = learningApprovedCandidate else { return }
        if let expiresAt = learningExpiresAt, expiresAt <= Date() {
            worldModelStatus = "This learning approval expired. Scan and review the app again."
            learningSessionID = nil
            learningApprovedCandidate = nil
            learningExpiresAt = nil
            showNextDecision()
            return
        }
        guard let data = try? JSONSerialization.data(withJSONObject: ["control_id": candidate.id]) else { return }
        worldModelBusy = true
        worldModelStatus = "Testing \(candidate.label) and restoring its exact previous value…"
        worldModel("run_learning_probe", id: sessionID, json: String(decoding: data, as: UTF8.self))
    }

    func stopLearningSession() {
        guard !worldModelBusy, let sessionID = learningSessionID else { return }
        worldModelBusy = true
        worldModelStatus = "Stopping the learning session…"
        worldModel("stop_learning_session", id: sessionID)
    }

    func discoverCurrentApplication() {
        guard !worldModelBusy else { return }
        worldModelBusy = true
        worldModelStatus = "Inspecting the current app without activating controls…"
        Task {
            await start()
            guard connectionState == .connected else {
                worldModelBusy = false
                worldModelStatus = errorMessage ?? "Sage Core is unavailable."
                return
            }
            var request = Sage_Ipc_V2_WorldModelCommand()
            request.operation = "discover_current_application"
            do {
                try await client.worldModel(request)
            } catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
            }
        }
    }

    func discoverPairedBrowser() {
        guard !worldModelBusy else { return }
        worldModelBusy = true
        worldModelStatus = "Inspecting the foreground paired browser tab without activating controls…"
        Task {
            await start()
            guard connectionState == .connected else {
                worldModelBusy = false
                worldModelStatus = errorMessage ?? "Sage Core is unavailable."
                return
            }
            var request = Sage_Ipc_V2_WorldModelCommand()
            request.operation = "discover_paired_browser"
            do {
                try await client.worldModel(request)
            } catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
            }
        }
    }

    func discoverLocalMediaRenderers() {
        guard !worldModelBusy else { return }
        upnpRendererCandidates = []
        upnpRendererTransportStates = [:]
        upnpRendererProtocolSummaries = [:]
        worldModelBusy = true
        worldModelStatus = "Searching the private local network for UPnP media renderers. Results are untrusted and cannot control devices."
        Task {
            await start()
            guard connectionState == .connected else {
                worldModelBusy = false
                worldModelStatus = errorMessage ?? "Sage Core is unavailable."
                return
            }
            var request = Sage_Ipc_V2_WorldModelCommand()
            request.operation = "discover_upnp_media_renderers"
            do {
                try await client.worldModel(request)
            } catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
            }
        }
    }

    func observeRendererTransport(_ candidate: RendererCandidateRecord) {
        guard !worldModelBusy else { return }
        worldModelBusy = true
        worldModelStatus = "Rechecking this renderer, then reading its reported transport state…"
        Task {
            await start()
            guard connectionState == .connected else {
                worldModelBusy = false
                worldModelStatus = errorMessage ?? "Sage Core is unavailable."
                return
            }
            var request = Sage_Ipc_V2_WorldModelCommand()
            request.operation = "observe_upnp_transport"
            var identity = Sage_Ipc_V2_UpnpRendererObservationTarget()
            identity.uniqueDeviceName = candidate.uniqueDeviceName
            identity.descriptionSha256 = candidate.descriptionSha256
            request.upnpRendererObservationTarget = identity
            do {
                try await client.worldModel(request)
            } catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
            }
        }
    }

    func observeRendererProtocolInfo(_ candidate: RendererCandidateRecord) {
        guard !worldModelBusy, candidate.connectionManagerServiceType != nil else { return }
        worldModelBusy = true
        worldModelStatus = "Rechecking this renderer, then reading its advertised receiver formats…"
        Task {
            await start()
            guard connectionState == .connected else {
                worldModelBusy = false
                worldModelStatus = errorMessage ?? "Sage Core is unavailable."
                return
            }
            var request = Sage_Ipc_V2_WorldModelCommand()
            request.operation = "observe_upnp_protocol_info"
            var identity = Sage_Ipc_V2_UpnpRendererObservationTarget()
            identity.uniqueDeviceName = candidate.uniqueDeviceName
            identity.descriptionSha256 = candidate.descriptionSha256
            request.upnpRendererObservationTarget = identity
            do {
                try await client.worldModel(request)
            } catch {
                worldModelBusy = false
                worldModelStatus = error.localizedDescription
            }
        }
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
            invalidateVoiceRepliesForNewInput()
            Task { await voiceInput.startFromMicrophoneButton() }
        }
    }

    func cancelVoiceInput() {
        voiceInput.cancelCurrentCommand()
    }

    func setWakeWordEnabled(_ enabled: Bool) {
        wakeWordEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "wakeWordEnabled")
        if enabled {
            Task { await voiceInput.enableWakeWordFromUserAction(phrase: wakePhrase) }
        } else {
            voiceInput.configureWakeWord(enabled: false, phrase: wakePhrase)
        }
    }

    func setWakePhrase(_ phrase: String) {
        let normalized = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        wakePhrase = normalized.isEmpty ? "Hey Sage" : normalized
        UserDefaults.standard.set(wakePhrase, forKey: "wakePhrase")
        voiceInput.updateWakePhrase(wakePhrase)
    }

    func stopSpokenReply() {
        stopCurrentSpokenReply(resumeWakeListening: true)
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

    func reconnect() {
        guard started, !reconnecting else { return }
        reconnecting = true
        connectionState = .starting
        Task {
            defer { reconnecting = false }
            do {
                do { try await client.connect() }
                catch {
                    let secret = try IPCSecretStore().loadOrCreateSecret()
                    try supervisor.startIfNeeded(secret: secret)
                    try await connectWithRetry()
                }
                guard started else { client.disconnect(); return }
                connectionState = .connected
                requestSnapshot(immediate: true)
                refreshConversationMessages()
            } catch {
                connectionState = .failed(error.localizedDescription)
                errorMessage = "Connection lost. Reconnect to check whether your request was accepted."
                isSubmitting = false
                if composerText.isEmpty, let unconfirmedRequest { composerText = unconfirmedRequest }
            }
        }
    }

    private func consume(_ event: Sage_Ipc_V2_CoreEvent) {
        switch event.event {
        case .intentPreview(let preview):
            guard preview.streamID == intentStreamID, preview.revision == intentRevision else { return }
            intentPreview = preview
            if voiceStreamActive,
               !voiceStreamPrefixSubmitted,
               !preview.streamedPrefix.isEmpty,
               preview.status == "prepared" {
                submit(
                    source: .voice,
                    requestOverride: preview.streamedPrefix,
                    streamedPrefix: true
                )
            }
            if preview.status == "paused" || preview.status == "stopping" { requestSnapshot() }
        case .decisionResolved(let resolution):
            removeDecision(resolution.decisionID)
        case .taskAccepted(let receipt):
            if streamedPrefixRequestIDs.remove(receipt.requestID) != nil {
                activeStreamedVoiceTaskID = receipt.taskID
                if let task = tasks.first(where: { $0.taskID == receipt.taskID }),
                   isFinished(task.status) {
                    activeStreamedVoiceTaskID = nil
                    voiceStreamPrefixSubmitted = false
                    streamedVoicePrefixText = nil
                    voiceStreamActive = false
                }
            }
            if streamedFinalRequestIDs.remove(receipt.requestID) != nil {
                voiceStreamActive = false
                voiceStreamPrefixSubmitted = false
                streamedVoicePrefixText = nil
                activeStreamedVoiceTaskID = nil
            }
            if let generation = voiceRequestGenerations.removeValue(forKey: receipt.requestID),
               generation == voiceTurnGeneration {
                voiceTaskGenerations[receipt.taskID] = generation
                voiceSentenceBuffers[receipt.taskID] = ResponseSentenceBuffer()
            }
            selectedTaskID = receipt.taskID
            isSubmitting = false
            draftActive = false
            if composerText == unconfirmedRequest { composerText = "" }
            unconfirmedRequest = nil
            requestSnapshot()
        case .stateSnapshot(let snapshot):
            snapshotRefresh?.received()
            refreshNotice = nil
            storageLocked = snapshot.storageLocked
            let decisions = snapshot.pendingApprovals.map { UserDecision.approval($0) }
                + snapshot.pendingQuestions.map { UserDecision.question($0) }
            decisionInbox = Dictionary(decisions.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            showNextDecision()
            tasks = snapshot.tasks
            stoppingTaskIDs = stoppingTaskIDs.filter { id in
                tasks.contains { $0.taskID == id && !isFinished($0.status) }
            }
            let retainedTaskIDs = Set(tasks.map(\.taskID))
            streamedResponses = streamedResponses.filter { retainedTaskIDs.contains($0.key) }
            if snapshot.hasKnowledge { applyKnowledge(snapshot.knowledge.json) }
            if let conversation = selectedConversationID,
               let latest = tasks.first(where: { $0.conversationID == conversation }),
               selectedTaskID != latest.taskID {
                selectedTaskID = latest.taskID
                timeline = timelinesByTaskID[latest.taskID] ?? []
                refreshConversationMessages()
            }
            if selectedTaskID == nil, !draftActive {
                selectedTaskID = visibleTasks.first?.taskID
                let conversationID = visibleTasks.first?.conversationID ?? ""
                selectedConversationID = conversationID.isEmpty ? nil : conversationID
                refreshConversationMessages()
                if let selectedTaskID {
                    timeline = timelinesByTaskID[selectedTaskID] ?? []
                }
            }
        case .taskUpdate(let task):
            let previousStatus = tasks.first(where: { $0.taskID == task.taskID })?.status
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
            if selectedTaskID == task.taskID, !task.conversationID.isEmpty {
                selectedConversationID = task.conversationID
            }
            if isFinished(task.status) {
                stoppingTaskIDs.remove(task.taskID)
                if task.taskID == activeStreamedVoiceTaskID {
                    activeStreamedVoiceTaskID = nil
                    voiceStreamPrefixSubmitted = false
                    streamedVoicePrefixText = nil
                    voiceStreamActive = false
                }
                if previousStatus != task.status { refreshConversationMessages() }
            }
            if [.failed, .cancelled, .interrupted].contains(task.status) {
                stopSpokenReplyIfNeeded(for: task.taskID)
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
            // Agent events are timeline deltas, not snapshot invalidations.
            // Only recover when an event references a task we have not received.
            if !agentEvent.taskID.isEmpty, !tasks.contains(where: { $0.taskID == agentEvent.taskID }) {
                requestSnapshot()
            }
        case .approvalRequest(let approval):
            let decision = UserDecision.approval(approval)
            decisionInbox[decision.id] = decision
            showNextDecision()
        case .questionRequest(let question):
            let decision = UserDecision.question(question)
            decisionInbox[decision.id] = decision
            showNextDecision()
        case .error(let error):
            if worldModelBusy {
                worldModelBusy = false
                worldModelStatus = error.message
            }
            if let goalRequest {
                goalRequestFailed(goalRequest, message: error.message)
            }
            if controllerRequest != nil {
                controllerDraftStatus = error.message
                controllerRequest = nil
            }
            if !error.requestID.isEmpty { voiceRequestGenerations.removeValue(forKey: error.requestID) }
            if streamedPrefixRequestIDs.remove(error.requestID) != nil {
                voiceStreamPrefixSubmitted = false
                streamedVoicePrefixText = nil
                voiceNotice = "Sage could not start the prepared step. Your speech remains a draft."
            }
            if streamedFinalRequestIDs.remove(error.requestID) != nil {
                voiceStreamActive = false
                if composerText.isEmpty, let unconfirmedRequest {
                    composerText = streamedVoicePrefixText
                        .flatMap { remainingVoiceRequest(after: $0, in: unconfirmedRequest) }
                        ?? unconfirmedRequest
                }
                unconfirmedRequest = nil
                voiceNotice = "The app step may already have run. Stop the held task, then continue from the remaining request."
            }
            if composerText.isEmpty, let unconfirmedRequest { composerText = unconfirmedRequest }
            unconfirmedRequest = nil
            if !error.taskID.isEmpty { stopSpokenReplyIfNeeded(for: error.taskID) }
            resolvingDecision = false
            isSubmitting = false
            errorMessage = error.message
            requestSnapshot()
        case .knowledgeState(let state):
            historyRefresh?.received()
            applyKnowledge(state.json)
        case .workflowState(let state):
            if let data = try? decodeKnowledge(WorkflowData.self, state.json) {
                skills = data.skills; workflows = data.workflows; schedules = data.schedules
                routineLearningEnabled = data.routineLearningEnabled ?? false
                routines = data.routines ?? []
                routineFamilies = data.routineFamilies ?? []
            }
        case .providerConnectionResult:
            break // Retained as a protocol-v2 compatibility event; Sage sends no provider probes.
        case .worldModelState(let state):
            worldModelSnapshotJSON = state.json
            if let data = state.json.data(using: .utf8),
               let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                applyGoalResult(result)
                if let candidates = decodeRendererCandidates(result["renderer_candidates"]) {
                    upnpRendererCandidates = candidates
                    upnpRendererTransportStates = [:]
                    upnpRendererProtocolSummaries = [:]
                    worldModelStatus = candidates.isEmpty
                        ? "No compatible UPnP media renderers answered the bounded private-LAN scan. No device was paired or controlled."
                        : "Found \(candidates.count) UPnP media renderer candidate\(candidates.count == 1 ? "" : "s"). They remain untrusted and cannot be controlled until pairing and policy support are implemented."
                }
                if let observation = decodeRendererTransportObservation(
                    result["transport_observation"]
                ) {
                    let state = observation.state.replacingOccurrences(of: "_", with: " ")
                    let status = observation.status.replacingOccurrences(of: "_", with: " ")
                    upnpRendererTransportStates[observation.uniqueDeviceName] =
                        "\(state) · \(status) · speed \(observation.currentSpeed)"
                    worldModelStatus = "Renderer reports \(state) (\(status), speed \(observation.currentSpeed)). This is an untrusted read-only observation; no playback command was sent."
                }
                if let observation = decodeRendererProtocolInfoObservation(
                    result["protocol_info_observation"]
                ) {
                    let formats = observation.sink
                        .map(\.contentFormat)
                        .reduce(into: [String]()) { values, format in
                            if !values.contains(format) { values.append(format) }
                        }
                    let summary = formats.isEmpty
                        ? "No receiver formats advertised"
                        : formats.prefix(8).joined(separator: ", ")
                    upnpRendererProtocolSummaries[observation.uniqueDeviceName] = summary
                    worldModelStatus = "Renderer advertises receiver formats: \(summary). This is untrusted compatibility evidence; no media was sent and a working path is not yet proven."
                }
                if let summaries = decodeControllerDrafts(result["controller_drafts"]) {
                    controllerDrafts = summaries
                }
                if let record = decodeControllerDraft(result["controller_draft"]) {
                    upsertControllerDraft(record)
                    switch controllerRequest {
                    case .compile(let taskID):
                        controllerDraftTaskID = taskID
                        controllerDraftStatus = "Saved a \(record.stepCount)-step unreviewed controller draft. Fresh review is still required; this draft grants no access."
                    case .inspect:
                        let decision = UserDecision.controllerDraft(record)
                        deferredDecisions.remove(decision.id)
                        decisionInbox[decision.id] = decision
                        showNextDecision()
                        NSApplication.shared.activate(ignoringOtherApps: true)
                    case .review:
                        let decision = UserDecision.controllerDraft(record)
                        decisionInbox[decision.id] = decision
                        pendingDecision = decision
                        controllerDraftStatus = result["fresh_rebind_verified"] as? Bool == true
                            ? "Every stored control matched the fresh interface. No action ran and no permission was granted."
                            : "Controller draft updated. No access was granted."
                    case .run:
                        break
                    case nil:
                        break
                    }
                    controllerRequest = nil
                }
                if case .run(let controllerID) = controllerRequest,
                   result["controller_id"] as? String == controllerID,
                   result["task_submitted"] as? Bool == true,
                   let taskID = result["task_id"] as? String {
                    controllerDraftStatus = "Started task \(taskID). Sage will request fresh approval before each controller action."
                    controllerRequest = nil
                    requestSnapshot(immediate: true)
                }
                if let details = result["safe_learning_candidate_details"] as? [[String: Any]],
                   let system = result["system"] as? [String: Any],
                   let systemID = system["id"] as? String,
                   let fingerprint = system["fingerprint"] as? String {
                    let systemLabel = system["label"] as? String ?? "current app"
                    learningCandidates = details.compactMap { detail in
                        guard let id = detail["id"] as? String,
                              let kind = detail["kind"] as? String,
                              let role = detail["role"] as? String,
                              let label = detail["label"] as? String else { return nil }
                        return LearningCandidate(
                            id: id,
                            kind: kind,
                            role: role,
                            label: label,
                            systemID: systemID,
                            systemFingerprint: fingerprint,
                            systemLabel: systemLabel
                        )
                    }
                    decisionInbox = decisionInbox.filter { key, value in
                        if case .learning = value { return false }
                        return true
                    }
                    for candidate in learningCandidates {
                        decisionInbox[candidate.decisionID] = .learning(candidate)
                    }
                    showNextDecision()
                    if worldModelBusy, let controls = result["observed_controls"] as? Int {
                        let candidateCount = learningCandidates.count
                        let hypotheses = result["passive_capability_hypotheses"] as? Int ?? 0
                        let skipped = result["skipped_controls"] as? Int ?? 0
                        let truncation = result["discovery_truncated"] as? Bool == true ? " The scan was bounded; inspect again to refresh the current interface." : ""
                        worldModelStatus = "\(systemLabel): \(controls) controls observed; \(skipped) omitted; \(hypotheses) non-executable hypotheses recorded (\(candidateCount) support a possible reversible test). No controls were changed.\(truncation)"
                    }
                }
                if worldModelBusy,
                   result["execution_available"] as? Bool == false,
                   result["learning_available"] as? Bool == false,
                   let controls = result["observed_controls"] as? Int,
                   let system = result["system"] as? [String: Any] {
                    let systemLabel = system["label"] as? String ?? "Paired browser origin"
                    let skipped = result["skipped_controls"] as? Int ?? 0
                    let truncation = result["truncated"] as? Bool == true ? " The scan reached its safety limit." : ""
                    worldModelStatus = "\(systemLabel): \(controls) visible controls recorded as private evidence; \(skipped) labels were omitted. No controls were activated or enabled for execution.\(truncation)"
                }
                if let session = result["session"] as? [String: Any],
                   let sessionID = session["id"] as? String {
                    learningSessionID = sessionID
                    if let candidate = learningApprovedCandidate {
                        decisionInbox.removeValue(forKey: candidate.decisionID)
                        showNextDecision()
                    }
                    if let expires = session["expires_at"] as? String {
                        let formatter = DateFormatter()
                        formatter.locale = Locale(identifier: "en_US_POSIX")
                        formatter.calendar = Calendar(identifier: .gregorian)
                        formatter.timeZone = TimeZone(secondsFromGMT: 0)
                        for suffix in [".SSSSSSSSSXXXXX", ".SSSXXXXX", "XXXXX"] {
                            formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss\(suffix)"
                            if let date = formatter.date(from: expires) {
                                learningExpiresAt = date
                                break
                            }
                        }
                    }
                    if let permitted = session["permitted_probes"] as? [String: String],
                       let controlID = permitted.keys.first,
                       let candidate = learningCandidates.first(where: { $0.id == controlID }) {
                        learningApprovedCandidate = candidate
                    }
                    let label = learningApprovedCandidate?.label ?? "approved control"
                    worldModelStatus = "Approved for \(label): one probe at a time, up to 20 probes or 10 minutes. Each probe must restore and verify the original value."
                }
                if result["probe_id"] is String, result["restoration_verified"] as? Bool == true {
                    let label = learningApprovedCandidate?.label ?? "approved control"
                    worldModelStatus = "Verified that \(label) changed and returned to its exact previous value. Sage recorded the three readbacks as local evidence."
                }
                if result["stopped"] as? Bool == true {
                    learningSessionID = nil
                    learningApprovedCandidate = nil
                    learningExpiresAt = nil
                    worldModelStatus = "Learning approval stopped."
                    showNextDecision()
                }
                if worldModelBusy { worldModelBusy = false }
            } else if let goalRequest {
                worldModelBusy = false
                goalRequestFailed(goalRequest, message: "Core returned an unreadable goal operation response.")
            }
        case .notification:
            isSubmitting = false
            requestSnapshot()
            refreshConversationMessages()
        case .modelResponseDelta(let delta):
            presentResponse(delta.text, taskID: delta.taskID, finished: delta.finished)
            if voiceTaskGenerations[delta.taskID] == voiceTurnGeneration {
                var sentenceBuffer = voiceSentenceBuffers[delta.taskID] ?? ResponseSentenceBuffer()
                let sentences = delta.finished
                    ? sentenceBuffer.finishCumulative(delta.text)
                    : sentenceBuffer.appendCumulative(delta.text)
                voiceSentenceBuffers[delta.taskID] = sentenceBuffer
                if !sentences.isEmpty {
                    beginSpeakingVoiceTask(delta.taskID, generation: voiceTurnGeneration)
                    for sentence in sentences { speechOutput.enqueue(sentence) }
                }
                if delta.finished {
                    if activeSpeechTaskID == delta.taskID {
                        speechOutput.finishStream()
                    } else {
                        voiceTaskGenerations.removeValue(forKey: delta.taskID)
                        voiceSentenceBuffers.removeValue(forKey: delta.taskID)
                    }
                }
            }
            if delta.finished { requestSnapshot(); refreshConversationMessages() }
        case .permissionRequest, nil:
            break
        }
    }

    private func presentResponse(_ text: String, taskID: String, finished: Bool) {
        pendingResponseText[taskID] = text
        if finished {
            flushResponsePresentation()
            return
        }
        guard responsePresentation == nil else { return }
        responsePresentation = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(50)) } catch { return }
            self?.flushResponsePresentation()
        }
    }

    private func flushResponsePresentation() {
        responsePresentation?.cancel()
        responsePresentation = nil
        for (taskID, text) in pendingResponseText {
            streamedResponses[taskID] = text
        }
        pendingResponseText.removeAll(keepingCapacity: true)
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
                voiceStreamActive = true
                voiceOverlay.show(phase: .listening, text: voiceTranscript)
            case .processing:
                voiceOverlay.show(phase: .processing, text: voiceTranscript)
            }
        }
        voiceInput.onTranscript = { [weak self] transcript in
            guard let self else { return }
            voiceTranscript = transcript
            if transcript.isEmpty {
                if voiceState == .idle || voiceState == .wakeListening {
                    cancelStreamedVoicePrefixIfNeeded()
                } else {
                    prepareIntent(transcript, voice: true)
                }
                return
            }
            voiceOverlay.show(phase: .listening, text: transcript)
            prepareIntent(transcript, voice: true)
            if VoiceReflex.parse(transcript) != nil { stopSpokenReply() }
        }
        voiceInput.onCommand = { [weak self] command, _ in
            guard let self else { return }
            voiceTranscript = command
            switch VoiceReflex.parse(command) {
            case .stop?, .hold?:
                prepareIntent(command, voice: true)
                voiceStreamActive = false
                return
            case .correction(let replacement)?:
                guard !replacement.isEmpty else { return }
                composerText = replacement
            case nil: composerText = command
            }
            voiceOverlay.show(phase: .processing, text: command)
            let finalizeStream = voiceStreamPrefixSubmitted
            voiceStreamActive = false
            submit(source: .voice, finalizeStream: finalizeStream)
        }
        voiceInput.onError = { [weak self] message in
            guard let self else { return }
            cancelStreamedVoicePrefixIfNeeded()
            voiceNotice = message
            voiceOverlay.show(phase: .error, text: message)
        }
        voiceInput.onWakeWordUnavailable = { [weak self] in
            guard let self else { return }
            wakeWordEnabled = false
            UserDefaults.standard.set(false, forKey: "wakeWordEnabled")
        }
        speechOutput.onStreamDrained = { [weak self] in
            guard let self else { return }
            guard let taskID = activeSpeechTaskID else { return }
            activeSpeechTaskID = nil
            isSpeaking = false
            voiceTaskGenerations.removeValue(forKey: taskID)
            voiceSentenceBuffers.removeValue(forKey: taskID)
            voiceInput.resumeWakeListeningAfterSpeech()
        }
    }

    private func voiceInteractionExecuting(_ transcript: String) {
        voiceOverlay.show(phase: .executing, text: transcript)
    }

    private func beginSpeakingVoiceTask(_ taskID: String, generation: Int) {
        guard generation == voiceTurnGeneration else { return }
        if activeSpeechTaskID != taskID {
            stopCurrentSpokenReply(resumeWakeListening: false)
            activeSpeechTaskID = taskID
            isSpeaking = true
            speechOutput.beginStream()
            voiceInput.pauseWakeListening()
        }
    }

    private func stopSpokenReplyIfNeeded(for taskID: String) {
        guard activeSpeechTaskID == taskID || voiceTaskGenerations[taskID] != nil else { return }
        if activeSpeechTaskID == taskID {
            stopCurrentSpokenReply(resumeWakeListening: true)
        } else {
            voiceTaskGenerations.removeValue(forKey: taskID)
            voiceSentenceBuffers.removeValue(forKey: taskID)
        }
    }

    private func stopCurrentSpokenReply(resumeWakeListening: Bool) {
        let taskID = activeSpeechTaskID
        speechOutput.stopImmediately()
        activeSpeechTaskID = nil
        isSpeaking = false
        if let taskID {
            voiceTaskGenerations.removeValue(forKey: taskID)
            voiceSentenceBuffers.removeValue(forKey: taskID)
        }
        if resumeWakeListening { voiceInput.resumeWakeListeningAfterSpeech() }
    }

    private func clearSpokenReplyState() {
        activeSpeechTaskID = nil
        isSpeaking = false
        voiceRequestGenerations.removeAll()
        voiceTaskGenerations.removeAll()
        voiceSentenceBuffers.removeAll()
    }

    private func invalidateVoiceRepliesForNewInput() {
        stopCurrentSpokenReply(resumeWakeListening: false)
        voiceTurnGeneration += 1
        voiceRequestGenerations.removeAll()
        voiceTaskGenerations.removeAll()
        voiceSentenceBuffers.removeAll()
    }

    private func persistTaskMetadata() {
        guard let data = try? JSONEncoder().encode(taskMetadata) else { return }
        UserDefaults.standard.set(data, forKey: Self.taskMetadataKey)
    }

    private func isFinished(_ status: Sage_Ipc_V2_TaskStatus) -> Bool {
        [.succeeded, .answered, .partial, .failed, .cancelled, .interrupted].contains(status)
    }

    private var selectedTaskIsActive: Bool {
        guard let selectedTaskID,
              let task = tasks.first(where: { $0.taskID == selectedTaskID }) else {
            return false
        }
        return !isFinished(task.status)
    }
}
