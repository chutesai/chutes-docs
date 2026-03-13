# Identity Provider API Reference

This section covers all endpoints related to identity provider.


## List Scopes

List all available OAuth2 scopes with descriptions.
This endpoint is public and can be used for documentation or scope selection UIs.


<div class="api-test-widget" data-widget-id="widget_get__idp_scopes"></div>
<script type="application/json" data-widget-config="widget_get__idp_scopes">{"endpoint":"/idp/scopes","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /idp/scopes`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Cli Login Nonce

Get a nonce for CLI-based hotkey signature login.


<div class="api-test-widget" data-widget-id="widget_get__idp_cli_login_nonce"></div>
<script type="application/json" data-widget-config="widget_get__idp_cli_login_nonce">{"endpoint":"/idp/cli_login/nonce","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /idp/cli_login/nonce`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Cli Login

CLI login endpoint for hotkey signature authentication.


<div class="api-test-widget" data-widget-id="widget_get__idp_cli_login"></div>
<script type="application/json" data-widget-config="widget_get__idp_cli_login">{"endpoint":"/idp/cli_login","method":"GET","parameters":[{"name":"hotkey","type":"string","required":true,"description":"","in":"query"},{"name":"signature","type":"string","required":true,"description":"","in":"query"},{"name":"nonce","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/cli_login`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string | Yes |  |
| signature | string | Yes |  |
| nonce | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## List Apps

List OAuth applications.

By default, returns apps owned by the current user, public apps, and apps shared with the user.
Set include_public=false to exclude public apps.
Set include_shared=false to exclude apps shared with the user.
Use search to filter by name or description.


<div class="api-test-widget" data-widget-id="widget_get__idp_apps"></div>
<script type="application/json" data-widget-config="widget_get__idp_apps">{"endpoint":"/idp/apps","method":"GET","requiresAuth":true,"parameters":[{"name":"include_public","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"include_shared","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"search","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"user_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/apps`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| include_public | boolean \| null | No |  |
| include_shared | boolean \| null | No |  |
| search | string \| null | No |  |
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| user_id | string \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Create App

Create a new OAuth application.


<div class="api-test-widget" data-widget-id="widget_post__idp_apps"></div>
<script type="application/json" data-widget-config="widget_post__idp_apps">{"endpoint":"/idp/apps","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"name":{"type":"string","title":"Name"},"description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Description"},"redirect_uris":{"items":{"type":"string"},"type":"array","title":"Redirect Uris"},"homepage_url":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Homepage Url"},"logo_url":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Url"},"public":{"type":"boolean","title":"Public","default":true},"refresh_token_lifetime_days":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Refresh Token Lifetime Days","default":30},"allowed_scopes":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Allowed Scopes"}},"required":["name","redirect_uris"]}}</script>

**Endpoint:** `POST /idp/apps`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes |  |
| description | string \| null | No |  |
| redirect_uris | string[] | Yes |  |
| homepage_url | string \| null | No |  |
| logo_url | string \| null | No |  |
| public | boolean | No |  |
| refresh_token_lifetime_days | integer \| null | No |  |
| allowed_scopes | string[] \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get App

Get details of an OAuth application.


<div class="api-test-widget" data-widget-id="widget_get__idp_apps__app_id_"></div>
<script type="application/json" data-widget-config="widget_get__idp_apps__app_id_">{"endpoint":"/idp/apps/{app_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/apps/{app_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Update App

Update an OAuth application.


<div class="api-test-widget" data-widget-id="widget_patch__idp_apps__app_id_"></div>
<script type="application/json" data-widget-config="widget_patch__idp_apps__app_id_">{"endpoint":"/idp/apps/{app_id}","method":"PATCH","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"name":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Name"},"description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Description"},"redirect_uris":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Redirect Uris"},"homepage_url":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Homepage Url"},"logo_url":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Url"},"active":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Active"},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public"},"refresh_token_lifetime_days":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Refresh Token Lifetime Days"},"allowed_scopes":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"title":"Allowed Scopes"}},"required":[]}}</script>

**Endpoint:** `PATCH /idp/apps/{app_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string \| null | No |  |
| description | string \| null | No |  |
| redirect_uris | string[] \| null | No |  |
| homepage_url | string \| null | No |  |
| logo_url | string \| null | No |  |
| active | boolean \| null | No |  |
| public | boolean \| null | No |  |
| refresh_token_lifetime_days | integer \| null | No |  |
| allowed_scopes | string[] \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete App

Delete an OAuth application.


<div class="api-test-widget" data-widget-id="widget_delete__idp_apps__app_id_"></div>
<script type="application/json" data-widget-config="widget_delete__idp_apps__app_id_">{"endpoint":"/idp/apps/{app_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /idp/apps/{app_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Regenerate App Secret

Regenerate the client secret for an OAuth application.


<div class="api-test-widget" data-widget-id="widget_post__idp_apps__app_id__regenerate_secret"></div>
<script type="application/json" data-widget-config="widget_post__idp_apps__app_id__regenerate_secret">{"endpoint":"/idp/apps/{app_id}/regenerate-secret","method":"POST","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /idp/apps/{app_id}/regenerate-secret`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Share App

Share an OAuth application with another user.


<div class="api-test-widget" data-widget-id="widget_post__idp_apps__app_id__share"></div>
<script type="application/json" data-widget-config="widget_post__idp_apps__app_id__share">{"endpoint":"/idp/apps/{app_id}/share","method":"POST","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"app_id_or_name":{"type":"string","title":"App Id Or Name"},"user_id_or_name":{"type":"string","title":"User Id Or Name"}},"required":["app_id_or_name","user_id_or_name"]}}</script>

**Endpoint:** `POST /idp/apps/{app_id}/share`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| app_id_or_name | string | Yes |  |
| user_id_or_name | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Unshare App

Remove sharing of an OAuth application with a user.


<div class="api-test-widget" data-widget-id="widget_delete__idp_apps__app_id__share__user_id_"></div>
<script type="application/json" data-widget-config="widget_delete__idp_apps__app_id__share__user_id_">{"endpoint":"/idp/apps/{app_id}/share/{user_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /idp/apps/{app_id}/share/{user_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| user_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List App Shares

List users an OAuth application is shared with.


<div class="api-test-widget" data-widget-id="widget_get__idp_apps__app_id__shares"></div>
<script type="application/json" data-widget-config="widget_get__idp_apps__app_id__shares">{"endpoint":"/idp/apps/{app_id}/shares","method":"GET","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/apps/{app_id}/shares`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Authorizations

List apps the current user has authorized.


<div class="api-test-widget" data-widget-id="widget_get__idp_authorizations"></div>
<script type="application/json" data-widget-config="widget_get__idp_authorizations">{"endpoint":"/idp/authorizations","method":"GET","requiresAuth":true,"parameters":[{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/authorizations`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Revoke App Authorization

Revoke authorization for an app.


<div class="api-test-widget" data-widget-id="widget_delete__idp_authorizations__app_id_"></div>
<script type="application/json" data-widget-config="widget_delete__idp_authorizations__app_id_">{"endpoint":"/idp/authorizations/{app_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"app_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /idp/authorizations/{app_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| app_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Authorize Get

OAuth2 Authorization Endpoint.
Displays login page if not authenticated, consent page if authenticated.
Checks for existing chutes-session-token cookie for SSO.


<div class="api-test-widget" data-widget-id="widget_get__idp_authorize"></div>
<script type="application/json" data-widget-config="widget_get__idp_authorize">{"endpoint":"/idp/authorize","method":"GET","parameters":[{"name":"response_type","type":"string","required":true,"description":"","in":"query"},{"name":"client_id","type":"string","required":true,"description":"","in":"query"},{"name":"redirect_uri","type":"string","required":true,"description":"","in":"query"},{"name":"scope","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"state","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"code_challenge","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"code_challenge_method","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/authorize`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| response_type | string | Yes |  |
| client_id | string | Yes |  |
| redirect_uri | string | Yes |  |
| scope | string \| null | No |  |
| state | string \| null | No |  |
| code_challenge | string \| null | No |  |
| code_challenge_method | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Login Post

Handle login form submission.


<div class="api-test-widget" data-widget-id="widget_post__idp_login"></div>
<script type="application/json" data-widget-config="widget_post__idp_login">{"endpoint":"/idp/login","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /idp/login`



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Authorize Consent Page

Show authorization consent page.


<div class="api-test-widget" data-widget-id="widget_get__idp_authorize_consent"></div>
<script type="application/json" data-widget-config="widget_get__idp_authorize_consent">{"endpoint":"/idp/authorize/consent","method":"GET","parameters":[{"name":"session_id","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /idp/authorize/consent`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Authorize Consent

Handle authorization consent form submission.


<div class="api-test-widget" data-widget-id="widget_post__idp_authorize_consent"></div>
<script type="application/json" data-widget-config="widget_post__idp_authorize_consent">{"endpoint":"/idp/authorize/consent","method":"POST","parameters":[{"name":"session_id","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `POST /idp/authorize/consent`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes |  |



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Token Endpoint

OAuth2 Token Endpoint.


<div class="api-test-widget" data-widget-id="widget_post__idp_token"></div>
<script type="application/json" data-widget-config="widget_post__idp_token">{"endpoint":"/idp/token","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /idp/token`



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Revoke Token Endpoint

OAuth2 Token Revocation Endpoint (RFC 7009).


<div class="api-test-widget" data-widget-id="widget_post__idp_token_revoke"></div>
<script type="application/json" data-widget-config="widget_post__idp_token_revoke">{"endpoint":"/idp/token/revoke","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /idp/token/revoke`



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Userinfo Endpoint

OpenID Connect UserInfo Endpoint.


<div class="api-test-widget" data-widget-id="widget_get__idp_userinfo"></div>
<script type="application/json" data-widget-config="widget_get__idp_userinfo">{"endpoint":"/idp/userinfo","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /idp/userinfo`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Introspect Token

OAuth2 Token Introspection Endpoint (RFC 7662).

Token format includes embedded token_id for O(1) lookup, so client auth is optional.

Allows clients to check if a token is still valid and get metadata about it.
Useful for determining if a user needs to re-authenticate.

Returns:
    - active: Whether the token is currently valid
    - exp: Expiration timestamp (Unix epoch)
    - iat: Issued at timestamp
    - scope: Space-separated list of scopes
    - client_id: The client that the token was issued to
    - username: The user's username
    - sub: The user's ID


<div class="api-test-widget" data-widget-id="widget_post__idp_token_introspect"></div>
<script type="application/json" data-widget-config="widget_post__idp_token_introspect">{"endpoint":"/idp/token/introspect","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /idp/token/introspect`



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
