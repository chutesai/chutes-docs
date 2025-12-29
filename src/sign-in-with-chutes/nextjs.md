# Sign in with Chutes: Next.js Guide

This guide walks you through implementing "Sign in with Chutes" OAuth in a Next.js application. By the end, your users will be able to authenticate with their Chutes account and your app can make API calls on their behalf.

## Quick Start with the Official SDK

The fastest way to add "Sign in with Chutes" to your Next.js app is using the official SDK repository with an AI coding assistant like Cursor:

**[github.com/chutesai/Sign-in-with-Chutes](https://github.com/chutesai/Sign-in-with-Chutes)**

Simply tell your AI assistant:

```
Add "Sign in with Chutes" to my Next.js app using the SDK at:
https://github.com/chutesai/Sign-in-with-Chutes
```

The AI will copy the integration files, set up routes, and configure your app automatically.

### Manual SDK Setup

Alternatively, use the setup wizard directly:

```bash
# Clone and set up
git clone https://github.com/chutesai/Sign-in-with-Chutes.git
cd Sign-in-with-Chutes
npm install

# Run the interactive setup wizard
npx tsx scripts/setup-chutes-app.ts

# Copy files from packages/nextjs/ to your project
```

The wizard will register your OAuth app and generate credentials.

---

The rest of this guide explains the implementation in detail if you want to understand how it works or customize the integration.

## Prerequisites

- Next.js 13+ with App Router
- A Chutes account with an API key
- Node.js 18+

## Installation

Install the required dependencies:

```bash
npm install
```

No additional OAuth libraries are required - this implementation uses native Web Crypto APIs and Next.js built-in features.

## OAuth App Registration

### Using the API

Register your OAuth application with Chutes:

```bash
curl -X POST "https://api.chutes.ai/idp/apps" \
  -H "Authorization: Bearer $CHUTES_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Next.js App",
    "description": "My application description",
    "redirect_uris": ["http://localhost:3000/api/auth/chutes/callback"],
    "homepage_url": "http://localhost:3000",
    "allowed_scopes": ["openid", "profile", "chutes:invoke"]
  }'
```

Save the returned `client_id` and `client_secret` for the next step.

**Important**: For production, add your production callback URL to `redirect_uris`:

```json
{
  "redirect_uris": [
    "http://localhost:3000/api/auth/chutes/callback",
    "https://yourapp.com/api/auth/chutes/callback"
  ]
}
```

## Environment Variables

Create a `.env.local` file in your project root:

```bash
# Required - OAuth Client Credentials
CHUTES_OAUTH_CLIENT_ID=cid_xxx
CHUTES_OAUTH_CLIENT_SECRET=csc_xxx

# Optional - Override default scopes
CHUTES_OAUTH_SCOPES="openid profile chutes:invoke"

# Optional - Explicitly set redirect URI (auto-detected if not set)
CHUTES_OAUTH_REDIRECT_URI=https://yourapp.com/api/auth/chutes/callback

# Optional - App URL for redirect URI construction
NEXT_PUBLIC_APP_URL=https://yourapp.com

# Optional - Override IDP base URL (rarely needed)
CHUTES_IDP_BASE_URL=https://api.chutes.ai
```

## Project Structure

Your authentication implementation will consist of these files:

```
app/
├── api/
│   └── auth/
│       └── chutes/
│           ├── login/
│           │   └── route.ts      # Initiates OAuth flow
│           ├── callback/
│           │   └── route.ts      # Handles OAuth callback
│           ├── logout/
│           │   └── route.ts      # Clears session
│           └── session/
│               └── route.ts      # Returns current session
lib/
├── chutesAuth.ts                 # Core OAuth utilities
└── serverAuth.ts                 # Server-side auth helpers
hooks/
└── useChutesSession.ts           # React hook for auth state
```

## Core Implementation

### OAuth Utilities (`lib/chutesAuth.ts`)

This file contains the core OAuth logic:

```typescript
import crypto from "crypto";

export interface OAuthConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  scopes: string[];
  idpBaseUrl: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface ChutesUser {
  sub: string;
  username: string;
  email?: string;
  name?: string;
}

// Get OAuth configuration from environment
export function getOAuthConfig(requestOrigin?: string): OAuthConfig {
  const clientId = process.env.CHUTES_OAUTH_CLIENT_ID;
  const clientSecret = process.env.CHUTES_OAUTH_CLIENT_SECRET;
  
  if (!clientId || !clientSecret) {
    throw new Error("Missing CHUTES_OAUTH_CLIENT_ID or CHUTES_OAUTH_CLIENT_SECRET");
  }

  const baseUrl = requestOrigin || 
    process.env.NEXT_PUBLIC_APP_URL || 
    "http://localhost:3000";
  
  const redirectUri = process.env.CHUTES_OAUTH_REDIRECT_URI || 
    `${baseUrl}/api/auth/chutes/callback`;

  const scopes = (process.env.CHUTES_OAUTH_SCOPES || "openid profile chutes:invoke")
    .split(" ");

  return {
    clientId,
    clientSecret,
    redirectUri,
    scopes,
    idpBaseUrl: process.env.CHUTES_IDP_BASE_URL || "https://api.chutes.ai",
  };
}

// Generate PKCE code verifier and challenge
export function generatePkce(): { verifier: string; challenge: string } {
  const verifier = crypto.randomBytes(32).toString("base64url");
  const challenge = crypto
    .createHash("sha256")
    .update(verifier)
    .digest("base64url");
  return { verifier, challenge };
}

// Generate random state for CSRF protection
export function generateState(): string {
  return crypto.randomBytes(16).toString("hex");
}

// Build the authorization URL
export function buildAuthorizeUrl(params: {
  state: string;
  codeChallenge: string;
  config: OAuthConfig;
}): string {
  const { state, codeChallenge, config } = params;
  
  const url = new URL(`${config.idpBaseUrl}/idp/authorize`);
  url.searchParams.set("client_id", config.clientId);
  url.searchParams.set("redirect_uri", config.redirectUri);
  url.searchParams.set("response_type", "code");
  url.searchParams.set("scope", config.scopes.join(" "));
  url.searchParams.set("state", state);
  url.searchParams.set("code_challenge", codeChallenge);
  url.searchParams.set("code_challenge_method", "S256");
  
  return url.toString();
}

// Exchange authorization code for tokens
export async function exchangeCodeForTokens(params: {
  code: string;
  codeVerifier: string;
  config: OAuthConfig;
}): Promise<TokenResponse> {
  const { code, codeVerifier, config } = params;

  const response = await fetch(`${config.idpBaseUrl}/idp/token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: config.clientId,
      client_secret: config.clientSecret,
      code,
      redirect_uri: config.redirectUri,
      code_verifier: codeVerifier,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Token exchange failed: ${error}`);
  }

  return response.json();
}

// Refresh expired tokens
export async function refreshTokens(params: {
  refreshToken: string;
  config: OAuthConfig;
}): Promise<TokenResponse> {
  const { refreshToken, config } = params;

  const response = await fetch(`${config.idpBaseUrl}/idp/token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      grant_type: "refresh_token",
      client_id: config.clientId,
      client_secret: config.clientSecret,
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    throw new Error("Token refresh failed");
  }

  return response.json();
}

// Fetch user info from Chutes
export async function fetchUserInfo(
  config: OAuthConfig,
  accessToken: string
): Promise<ChutesUser> {
  const response = await fetch(`${config.idpBaseUrl}/idp/userinfo`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });

  if (!response.ok) {
    throw new Error("Failed to fetch user info");
  }

  return response.json();
}
```

### Server-Side Helpers (`lib/serverAuth.ts`)

Helper functions for accessing auth state on the server:

```typescript
import { cookies } from "next/headers";
import type { ChutesUser } from "./chutesAuth";

const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
};

// Get access token from cookies
export async function getServerAccessToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get("chutes_access_token")?.value || null;
}

// Get refresh token from cookies
export async function getServerRefreshToken(): Promise<string | null> {
  const cookieStore = await cookies();
  return cookieStore.get("chutes_refresh_token")?.value || null;
}

// Get cached user info from cookies
export async function getServerUserInfo(): Promise<ChutesUser | null> {
  const cookieStore = await cookies();
  const userCookie = cookieStore.get("chutes_user")?.value;
  
  if (!userCookie) return null;
  
  try {
    return JSON.parse(userCookie);
  } catch {
    return null;
  }
}

// Check if user is authenticated
export async function isAuthenticated(): Promise<boolean> {
  const token = await getServerAccessToken();
  return !!token;
}

// Set auth cookies (for use in route handlers)
export function setAuthCookies(
  headers: Headers,
  tokens: { access_token: string; refresh_token: string },
  user: ChutesUser
): void {
  const cookieOptions = `; HttpOnly; ${
    process.env.NODE_ENV === "production" ? "Secure; " : ""
  }SameSite=Lax; Path=/`;

  headers.append(
    "Set-Cookie",
    `chutes_access_token=${tokens.access_token}${cookieOptions}`
  );
  headers.append(
    "Set-Cookie",
    `chutes_refresh_token=${tokens.refresh_token}${cookieOptions}`
  );
  headers.append(
    "Set-Cookie",
    `chutes_user=${JSON.stringify(user)}${cookieOptions}`
  );
}

// Clear auth cookies (for logout)
export function clearAuthCookies(headers: Headers): void {
  const expiredOptions = "; HttpOnly; Path=/; Max-Age=0";
  headers.append("Set-Cookie", `chutes_access_token=${expiredOptions}`);
  headers.append("Set-Cookie", `chutes_refresh_token=${expiredOptions}`);
  headers.append("Set-Cookie", `chutes_user=${expiredOptions}`);
  headers.append("Set-Cookie", `chutes_state=${expiredOptions}`);
  headers.append("Set-Cookie", `chutes_verifier=${expiredOptions}`);
}
```

### Login Route (`app/api/auth/chutes/login/route.ts`)

Initiates the OAuth flow:

```typescript
import { NextResponse } from "next/server";
import {
  getOAuthConfig,
  generatePkce,
  generateState,
  buildAuthorizeUrl,
} from "@/lib/chutesAuth";

export async function GET(request: Request) {
  const origin = new URL(request.url).origin;
  const config = getOAuthConfig(origin);
  
  // Generate PKCE and state
  const { verifier, challenge } = generatePkce();
  const state = generateState();
  
  // Build authorization URL
  const authorizeUrl = buildAuthorizeUrl({
    state,
    codeChallenge: challenge,
    config,
  });
  
  // Create response with redirect
  const response = NextResponse.redirect(authorizeUrl);
  
  // Store state and verifier in cookies for callback validation
  const cookieOptions = `; HttpOnly; ${
    process.env.NODE_ENV === "production" ? "Secure; " : ""
  }SameSite=Lax; Path=/; Max-Age=600`;
  
  response.headers.append("Set-Cookie", `chutes_state=${state}${cookieOptions}`);
  response.headers.append("Set-Cookie", `chutes_verifier=${verifier}${cookieOptions}`);
  
  return response;
}
```

### Callback Route (`app/api/auth/chutes/callback/route.ts`)

Handles the OAuth callback:

```typescript
import { NextResponse, type NextRequest } from "next/server";
import { cookies } from "next/headers";
import {
  getOAuthConfig,
  exchangeCodeForTokens,
  fetchUserInfo,
} from "@/lib/chutesAuth";
import { setAuthCookies } from "@/lib/serverAuth";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const error = searchParams.get("error");

  // Handle OAuth errors
  if (error) {
    return NextResponse.redirect(
      new URL(`/?error=${encodeURIComponent(error)}`, request.url)
    );
  }

  // Validate required parameters
  if (!code || !state) {
    return NextResponse.redirect(
      new URL("/?error=missing_params", request.url)
    );
  }

  // Get stored state and verifier from cookies
  const cookieStore = await cookies();
  const storedState = cookieStore.get("chutes_state")?.value;
  const codeVerifier = cookieStore.get("chutes_verifier")?.value;

  // Validate state to prevent CSRF
  if (!storedState || state !== storedState) {
    return NextResponse.redirect(
      new URL("/?error=invalid_state", request.url)
    );
  }

  if (!codeVerifier) {
    return NextResponse.redirect(
      new URL("/?error=missing_verifier", request.url)
    );
  }

  try {
    const origin = new URL(request.url).origin;
    const config = getOAuthConfig(origin);

    // Exchange code for tokens
    const tokens = await exchangeCodeForTokens({
      code,
      codeVerifier,
      config,
    });

    // Fetch user info
    const user = await fetchUserInfo(config, tokens.access_token);

    // Create response with redirect to home
    const response = NextResponse.redirect(new URL("/", request.url));

    // Set auth cookies
    setAuthCookies(response.headers, tokens, user);

    // Clear temporary cookies
    response.headers.append(
      "Set-Cookie",
      "chutes_state=; HttpOnly; Path=/; Max-Age=0"
    );
    response.headers.append(
      "Set-Cookie",
      "chutes_verifier=; HttpOnly; Path=/; Max-Age=0"
    );

    return response;
  } catch (error) {
    console.error("OAuth callback error:", error);
    return NextResponse.redirect(
      new URL("/?error=auth_failed", request.url)
    );
  }
}
```

### Logout Route (`app/api/auth/chutes/logout/route.ts`)

Clears the user's session:

```typescript
import { NextResponse } from "next/server";
import { clearAuthCookies } from "@/lib/serverAuth";

export async function POST(request: Request) {
  const response = NextResponse.redirect(new URL("/", request.url));
  clearAuthCookies(response.headers);
  return response;
}

// Also support GET for convenience
export async function GET(request: Request) {
  return POST(request);
}
```

### Session Route (`app/api/auth/chutes/session/route.ts`)

Returns the current session state:

```typescript
import { NextResponse } from "next/server";
import {
  getServerAccessToken,
  getServerUserInfo,
} from "@/lib/serverAuth";

export async function GET() {
  const token = await getServerAccessToken();
  const user = await getServerUserInfo();

  if (!token || !user) {
    return NextResponse.json({ isSignedIn: false, user: null });
  }

  return NextResponse.json({ isSignedIn: true, user });
}
```

### React Hook (`hooks/useChutesSession.ts`)

Client-side hook for accessing auth state:

```typescript
"use client";

import { useState, useEffect, useCallback } from "react";

interface ChutesUser {
  sub: string;
  username: string;
  email?: string;
  name?: string;
}

interface SessionState {
  isSignedIn: boolean;
  user: ChutesUser | null;
  loading: boolean;
  loginUrl: string;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
}

export function useChutesSession(): SessionState {
  const [isSignedIn, setIsSignedIn] = useState(false);
  const [user, setUser] = useState<ChutesUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/auth/chutes/session");
      const data = await response.json();
      setIsSignedIn(data.isSignedIn);
      setUser(data.user);
    } catch (error) {
      console.error("Failed to fetch session:", error);
      setIsSignedIn(false);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await fetch("/api/auth/chutes/logout", { method: "POST" });
      setIsSignedIn(false);
      setUser(null);
    } catch (error) {
      console.error("Logout failed:", error);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    isSignedIn,
    user,
    loading,
    loginUrl: "/api/auth/chutes/login",
    refresh,
    logout,
  };
}
```

## Usage Examples

### Sign In Button Component

```tsx
"use client";

import { useChutesSession } from "@/hooks/useChutesSession";

export function AuthButton() {
  const { isSignedIn, user, loading, loginUrl, logout } = useChutesSession();

  if (loading) {
    return <button disabled>Loading...</button>;
  }

  if (isSignedIn && user) {
    return (
      <div>
        <span>Welcome, {user.username}!</span>
        <button onClick={logout}>Sign Out</button>
      </div>
    );
  }

  return (
    <a href={loginUrl}>
      Sign in with Chutes
    </a>
  );
}
```

### Protected Server Component

```tsx
import { redirect } from "next/navigation";
import { isAuthenticated, getServerUserInfo } from "@/lib/serverAuth";

export default async function DashboardPage() {
  const authenticated = await isAuthenticated();
  
  if (!authenticated) {
    redirect("/api/auth/chutes/login");
  }

  const user = await getServerUserInfo();

  return (
    <div>
      <h1>Dashboard</h1>
      <p>Welcome, {user?.username}!</p>
    </div>
  );
}
```

### Custom Post-Login Redirect

Modify the callback route to redirect to a specific page:

```typescript
// In callback/route.ts
const response = NextResponse.redirect(new URL("/dashboard", request.url));
```

Or redirect to where the user was before:

```typescript
// Store the return URL before login
const returnTo = cookieStore.get("return_to")?.value || "/";
const response = NextResponse.redirect(new URL(returnTo, request.url));
```

## Advanced Usage

### Token Refresh

Access tokens expire after approximately 1 hour. Implement token refresh:

```typescript
import {
  getServerAccessToken,
  getServerRefreshToken,
} from "@/lib/serverAuth";
import { refreshTokens, getOAuthConfig } from "@/lib/chutesAuth";

async function getValidToken(): Promise<string | null> {
  const token = await getServerAccessToken();
  
  if (token) {
    return token;
  }

  // Try to refresh
  const refreshToken = await getServerRefreshToken();
  if (!refreshToken) {
    return null;
  }

  try {
    const config = getOAuthConfig();
    const newTokens = await refreshTokens({ refreshToken, config });
    // Note: You'll need to set new cookies in a route handler
    return newTokens.access_token;
  } catch {
    return null;
  }
}
```

### Middleware Protection

Protect routes with Next.js middleware:

```typescript
// middleware.ts
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const token = request.cookies.get("chutes_access_token");

  // Protect /dashboard routes
  if (request.nextUrl.pathname.startsWith("/dashboard")) {
    if (!token) {
      return NextResponse.redirect(
        new URL("/api/auth/chutes/login", request.url)
      );
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*"],
};
```

### Using with Vercel AI SDK

Make AI calls using the user's token for billing:

```typescript
import { createChutes } from "@chutes-ai/ai-sdk-provider";
import { generateText, streamText } from "ai";
import { getServerAccessToken } from "@/lib/serverAuth";

export async function POST(req: Request) {
  const token = await getServerAccessToken();

  if (!token) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Use the user's access token instead of your API key
  const chutes = createChutes({ apiKey: token });
  const { message } = await req.json();

  const { text } = await generateText({
    model: chutes("deepseek-ai/DeepSeek-V3-0324"),
    prompt: message,
  });

  return Response.json({ text });
}
```

For streaming responses:

```typescript
import { createChutes } from "@chutes-ai/ai-sdk-provider";
import { streamText } from "ai";
import { getServerAccessToken } from "@/lib/serverAuth";

export async function POST(req: Request) {
  const token = await getServerAccessToken();

  if (!token) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const chutes = createChutes({ apiKey: token });
  const { message } = await req.json();

  const result = await streamText({
    model: chutes("meta-llama/Llama-3.1-70B-Instruct"),
    prompt: message,
  });

  return result.toDataStreamResponse();
}
```

## Security Best Practices

### 1. Keep Secrets Server-Side

Never expose `CHUTES_OAUTH_CLIENT_SECRET` to the client. All token operations happen in API routes.

### 2. Use HttpOnly Cookies

All auth cookies are set with `httpOnly: true` to prevent XSS attacks from accessing tokens.

### 3. Validate State Parameter

Always validate the `state` parameter in the callback to prevent CSRF attacks.

### 4. Use PKCE

PKCE prevents authorization code interception. The implementation handles this automatically.

### 5. HTTPS in Production

Cookies are set with `secure: true` in production, requiring HTTPS.

### 6. Limit Scope Requests

Only request the scopes you actually need:

```bash
# Good - minimal scopes
CHUTES_OAUTH_SCOPES="openid profile chutes:invoke"

# Avoid requesting unnecessary scopes
CHUTES_OAUTH_SCOPES="openid profile chutes:invoke billing:read account:read"
```

### 7. Handle Token Expiry

Implement token refresh or prompt users to re-authenticate when tokens expire.

## Troubleshooting

### "Missing client credentials" Error

Ensure environment variables are set correctly:

```bash
echo $CHUTES_OAUTH_CLIENT_ID
echo $CHUTES_OAUTH_CLIENT_SECRET
```

### "Invalid state" Error

This occurs when the state cookie is missing or doesn't match. Causes:
- Cookies blocked by browser
- Session expired (cookies expire after 10 minutes)
- Multiple login attempts in different tabs

### "Token exchange failed" Error

Check that:
- `redirect_uri` matches exactly what's registered with your OAuth app
- `client_secret` is correct
- The authorization code hasn't expired (codes are single-use)

### Cookies Not Being Set

Ensure your callback URL matches the domain where cookies are set. In development, use `http://localhost:3000` consistently.

## Next Steps

- Review the [Sign in with Chutes Overview](overview) for OAuth concepts
- Explore the [Vercel AI SDK Integration](/docs/integrations/vercel-ai-sdk) for AI features
- Join our [Discord community](https://discord.gg/wHrXwWkCRz) for support

