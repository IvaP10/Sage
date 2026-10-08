import Foundation

/// Presentation is derived from broker facts. A model answer and a cancelled
/// request must never be displayed as proof that effects were completed or undone.
enum TaskPresentation {
    static func isFinished(_ status: Sage_Ipc_V2_TaskStatus) -> Bool {
        [.succeeded, .answered, .partial, .failed, .cancelled, .interrupted].contains(status)
    }

    static func label(_ status: Sage_Ipc_V2_TaskStatus) -> String {
        switch status {
        case .pending: "Queued"
        case .planning: "Thinking"
        case .running: "Working"
        case .waitingForApproval: "Approval needed"
        case .waitingForUser: "Your answer is needed"
        case .paused: "Paused"
        case .succeeded: "Actions verified"
        case .answered: "Answered"
        case .partial: "Partially completed"
        case .failed: "Could not finish"
        case .cancelled: "Stopped"
        case .interrupted: "Interrupted"
        default: "Waiting for state"
        }
    }

    static func response(for task: Sage_Ipc_V2_TaskUpdate, messages: [ConversationMessage], streamed: String?) -> String? {
        if messages.contains(where: { $0.taskId == task.taskID && $0.role == "assistant" }) { return nil }
        let text = isFinished(task.status) && !task.finalOutcome.isEmpty
            ? task.finalOutcome : (streamed?.isEmpty == false ? streamed! : task.finalOutcome)
        return text.isEmpty ? nil : text
    }

    static func verifiedCount(_ task: Sage_Ipc_V2_TaskUpdate) -> UInt32 {
        task.hasExecutionFacts ? task.executionFacts.verified : task.completedActions
    }

    static func progress(_ task: Sage_Ipc_V2_TaskUpdate) -> Double? {
        guard task.totalActions > 0 else { return nil }
        return min(1, Double(verifiedCount(task)) / Double(task.totalActions))
    }

    static func needsAttention(_ task: Sage_Ipc_V2_TaskUpdate) -> Bool {
        [.waitingForApproval, .waitingForUser, .paused, .interrupted, .partial, .failed].contains(task.status)
            || task.executionFacts.uncertain > 0
            || task.undoState == "uncertain"
    }
}
