import Foundation

struct ConversationRecord: Decodable, Identifiable {
    let id: String
    let title: String
    let summary: String
    let pinned: Bool
    let archived: Bool
}

struct ConversationMessage: Decodable, Identifiable {
    let id: String
    let conversationId: String
    let taskId: String?
    let role: String
    let content: String
}

struct MemoryRecord: Decodable, Identifiable {
    let id: String
    let kind: String
    let subject: String
    let content: String
    let confidence: Double
    let enabled: Bool
}

struct KnowledgeData: Decodable {
    let conversations: [ConversationRecord]
    let messages: [ConversationMessage]
    let memories: [MemoryRecord]
    let memoryEnabled: Bool
}

struct SkillRecord: Decodable, Identifiable {
    let id: String; let name: String; let description: String; let enabled: Bool
    let sourcePaused: Bool
    let preview: String; let reviewDigestCandidate: String
}
struct WorkflowRecord: Decodable, Identifiable {
    let id: String; let name: String; let enabled: Bool; let sourcePaused: Bool
}
struct ScheduleRecord: Decodable, Identifiable {
    let id: String; let name: String; let request: String; let enabled: Bool
    let sourcePaused: Bool; let nextRunAt: String; let lastError: String?
}
struct RoutineRecord: Decodable, Identifiable {
    let id: String
    let requests: [String]
    let steps: [String]
    let verifiedRuns: Int
    let ready: Bool
    let enabled: Bool
    let reviewDigest: String
    let evolution: RoutineEvolutionRecord?
}
struct RoutineEvolutionRecord: Decodable {
    let sourceRequest: String
    let sourceVerifiedRuns: Int
    let unchangedEffectClasses: [String]
    let addedEffectClasses: [String]
    let removedEffectClasses: [String]
}
struct RoutineBranchRecord: Decodable, Identifiable {
    let routineId: String
    let requests: [String]
    let nextSteps: [String]
    let verifiedRuns: Int
    var id: String { routineId }
}
struct RoutineFamilyRecord: Decodable, Identifiable {
    let id: String
    let reviewDigest: String
    let sharedSteps: [String]
    let branches: [RoutineBranchRecord]
    let verifiedRuns: Int
}
struct ControllerDraftStep: Decodable, Identifiable {
    let id: String
    let controlName: String
    let role: String
    let expectedEffect: String
    let verification: String
    let restoration: String?
}
struct ControllerDraftRecord: Decodable, Identifiable {
    let id: String
    let systemId: String
    let systemLabel: String
    let revision: UInt64
    let status: String
    let stepCount: Int
    let taskId: String?
    let steps: [ControllerDraftStep]?
}
struct RendererCandidateRecord: Decodable, Identifiable {
    let source: String
    let uniqueDeviceName: String
    let deviceType: String
    let friendlyName: String
    let manufacturer: String?
    let modelName: String?
    let avTransportServiceType: String
    let connectionManagerServiceType: String?
    let descriptionSha256: String
    let observedAt: String
    let evidenceScope: String

    var id: String { uniqueDeviceName }
}
struct RendererTransportObservationRecord: Decodable {
    let uniqueDeviceName: String
    let state: String
    let status: String
    let currentSpeed: String
}
struct RendererProtocolInfoEntryRecord: Decodable {
    let `protocol`: String
    let network: String
    let contentFormat: String
    let additionalInfo: String
}
struct RendererProtocolInfoObservationRecord: Decodable {
    let uniqueDeviceName: String
    let source: [RendererProtocolInfoEntryRecord]
    let sink: [RendererProtocolInfoEntryRecord]
}
struct WorkflowData: Decodable {
    let skills: [SkillRecord]; let workflows: [WorkflowRecord]; let schedules: [ScheduleRecord]
    let routineLearningEnabled: Bool?
    let routines: [RoutineRecord]?
    let routineFamilies: [RoutineFamilyRecord]?
}

func decodeKnowledge<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: Data(json.utf8))
}
