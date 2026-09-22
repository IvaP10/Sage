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
    let preview: String; let reviewDigestCandidate: String
}
struct WorkflowRecord: Decodable, Identifiable { let id: String; let name: String; let enabled: Bool }
struct ScheduleRecord: Decodable, Identifiable {
    let id: String; let name: String; let request: String; let enabled: Bool; let nextRunAt: String; let lastError: String?
}
struct WorkflowData: Decodable { let skills: [SkillRecord]; let workflows: [WorkflowRecord]; let schedules: [ScheduleRecord] }

func decodeKnowledge<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: Data(json.utf8))
}
