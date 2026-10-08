import Foundation

enum SageClientError: LocalizedError {
    case connectionFailed(String)
    case authenticationFailed(String)
    case protocolError(String)

    var errorDescription: String? {
        switch self {
        case .connectionFailed(let message), .authenticationFailed(let message), .protocolError(let message):
            message
        }
    }
}
