# Miscellaneous API Reference

This section covers all endpoints related to miscellaneous.


## Proxy


<div class="api-test-widget" data-widget-id="widget_get__misc_proxy"></div>
<script type="application/json" data-widget-config="widget_get__misc_proxy">{"endpoint":"/misc/proxy","method":"GET","parameters":[{"name":"url","type":"string","required":true,"description":"","in":"query"},{"name":"stream","type":"boolean","required":false,"description":"Stream the response for large files/videos","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /misc/proxy`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | Yes |  |
| stream | boolean | No | Stream the response for large files/videos |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Hf Repo Info

Proxy endpoint for HF repo file info.


<div class="api-test-widget" data-widget-id="widget_get__misc_hf_repo_info"></div>
<script type="application/json" data-widget-config="widget_get__misc_hf_repo_info">{"endpoint":"/misc/hf_repo_info","method":"GET","parameters":[{"name":"repo_id","type":"string","required":true,"description":"","in":"query"},{"name":"repo_type","type":"string","required":false,"description":"","in":"query"},{"name":"revision","type":"string","required":false,"description":"","in":"query"},{"name":"hf_token","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /misc/hf_repo_info`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| repo_id | string | Yes |  |
| repo_type | string | No |  |
| revision | string | No |  |
| hf_token | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
