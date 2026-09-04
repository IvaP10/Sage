using Windows.Security.Credentials.UI;

namespace Sage.Windows;

internal static class NativeAuthentication
{
    public static async Task<bool> AuthenticateAsync(string reason)
    {
        var availability = await UserConsentVerifier.CheckAvailabilityAsync();
        if (availability != UserConsentVerifierAvailability.Available)
        {
            throw new InvalidOperationException($"Windows device authentication is unavailable: {availability}");
        }
        var result = await UserConsentVerifier.RequestVerificationAsync(reason);
        return result == UserConsentVerificationResult.Verified;
    }
}
