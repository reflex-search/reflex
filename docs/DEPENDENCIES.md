# Dependency Tracking in Reflex

**Status:** Implemented (`rfx deps`, `rfx analyze`, `rfx query --dependencies`, MCP dependency tools)
**Last Updated:** 2026-09-28 (Reflex 2.0.3)

Reflex extracts **static imports only** (string literals); dynamic imports are filtered out
by design, so the same codebase always gives the same graph.

## Overview

Reflex tracks **file dependencies** to help developers and AI coding agents understand relationships between files in a codebase. This feature enables:

- **Impact Analysis**: "What breaks if I change this file?"
- **Context Understanding**: "What does this code depend on?"
- **Reverse Lookup**: "What uses this file/module?"
- **Architecture Analysis**: Finding circular dependencies, hotspots, and orphaned files

### Key Design Principles

1. **Index Shallow, Traverse Deep**: Store only direct (depth-1) dependencies; compute deeper relationships on demand
2. **Three entry points**: search augmentation (`rfx query --dependencies`), one file (`rfx deps`), and the whole graph (`rfx analyze`)
3. **Lazy Evaluation**: Compute expensive operations only when explicitly requested
4. **Cross-Language Consistency**: Single API works across the 15 supported code languages
5. **AI-First Design**: Structured output optimized for AI coding agents

---

## Motivation

### Problem: AI Agents Need Dependency Context

When AI coding agents work with code, they need to understand:
- What a file imports (to understand its context)
- What imports a file (to assess change impact)
- Whether code is safe to delete (no incoming dependencies)
- How modules are interconnected (architectural understanding)

### Why Not Use grep/awk?

Traditional tools have limitations:
- **Slow on large trees**: a full scan on every query
- **Inaccurate**: False positives from comments, strings, logs
- **Language-Specific**: Different regex per language (error-prone)
- **No Resolution**: Returns raw import strings, not resolved paths
- **Manual Parsing**: Agent must parse and classify results

### Reflex Advantages

- **Indexed**: dependencies are stored at index time, so lookups do not scan files
- **Accurate**: Tree-sitter parsing eliminates false positives
- **Universal**: Same API across all 15 code languages
- **Resolved**: Import paths resolved to actual files
- **Structured**: Clean JSON output, no manual parsing

---

## Architecture

### Core Concept: Depth-1 Storage

Reflex stores **only direct dependencies** (depth-1) in the database. Deeper relationships are computed on-demand by traversing the index.

```
Example Codebase:
  file_a.rs → [file_b.rs, file_c.rs]     (stored)
  file_b.rs → [file_d.rs, file_e.rs]     (stored)
  file_d.rs → [file_f.rs]                 (stored)

Depth-3 Query from file_a.rs:
  1. Lookup file_a.rs → get [file_b.rs, file_c.rs]
  2. Lookup file_b.rs → get [file_d.rs, file_e.rs]
  3. Lookup file_d.rs → get [file_f.rs]

Result: 6 lookups, all indexed (O(1) each)
```

### Why This Works

- **Storage**: O(n) instead of O(n²) for full closure
- **Updates**: Change one file, update one row
- **Flexibility**: Compute any depth on-demand
- **Performance**: Most files have 5-20 deps, not hundreds

### Component Integration

```
Indexer (src/indexer.rs)
  → DependencyExtractor (src/parsers/mod.rs): imports from the tree-sitter AST, per language
  → PathResolver (src/dependency.rs): in-memory import → file resolution
  → DependencyWriter (src/dependency.rs): one transaction into meta.db
DependencyIndex (src/dependency.rs): lookups, transitive traversal, hotspots, unused files, islands, cycles
```

---
## Storage Schema

### Database Tables

#### file_dependencies Table

```sql
CREATE TABLE file_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,           -- Source file (FK to files.id)
    imported_path TEXT NOT NULL,        -- Import as written in source
    resolved_file_id INTEGER,           -- Target file (FK to files.id), NULL if external
    import_type TEXT NOT NULL,          -- 'internal', 'external', 'stdlib'
    line_number INTEGER NOT NULL,       -- Line where import appears
    imported_symbols TEXT,              -- JSON array of specific symbols (if selective import)
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY (resolved_file_id) REFERENCES files(id) ON DELETE SET NULL
);

CREATE INDEX idx_deps_file ON file_dependencies(file_id);
CREATE INDEX idx_deps_resolved ON file_dependencies(resolved_file_id);
CREATE INDEX idx_deps_type ON file_dependencies(import_type);
```

### Import Types

- **internal**: Import references another file in the project
- **external**: Import from external package/library (node_modules, pip, cargo, etc.)
- **stdlib**: Standard library import (depends on language)

### Example Data

```sql
-- Rust: use std::collections::HashMap;
INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
VALUES (42, 'std::collections::HashMap', NULL, 'stdlib', 5, '["HashMap"]');

-- Rust: use crate::models::User;
INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
VALUES (42, 'crate::models::User', 15, 'internal', 7, '["User"]');

-- Python: import requests
INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
VALUES (55, 'requests', NULL, 'external', 3, NULL);

-- Python: from .utils import format_date
INSERT INTO file_dependencies (file_id, imported_path, resolved_file_id, import_type, line_number, imported_symbols)
VALUES (55, '.utils', 62, 'internal', 5, '["format_date"]');
```

---

## Command Interface

### `rfx query --dependencies` (search augmentation)

Adds each result file's imports to the search output.

```bash
rfx query "ApiClient" --dependencies
```

### `rfx deps <file>` (one file)

```bash
rfx deps src/main.rs                  # what this file imports
rfx deps src/config.rs --reverse      # what imports this file
rfx deps src/api.rs --depth 3         # transitive dependencies
rfx deps src/main.rs --format table   # formats: tree (default), table
rfx deps src/main.rs --json --pretty
```

### `rfx analyze` (whole graph)

```bash
rfx analyze --circular                      # circular dependencies
rfx analyze --hotspots --min-dependents 5   # most-imported files
rfx analyze --unused                        # files nothing imports
rfx analyze --islands                       # disconnected components
rfx analyze --circular --glob "src/**"      # limit to a subtree
rfx analyze --hotspots --count              # count only
```

Output: `--format tree|table`, `--json`, `--pretty`; paging with `--limit`, `--offset`, `--all`;
`--sort asc|desc`.

### MCP tools

`get_dependencies`, `get_dependents`, `get_transitive_deps`, `find_hotspots`, and the
structural tools `find_circular`, `find_islands`, `find_unused`, `analyze_summary`.
See `docs/mcp-tool-cheatsheet.md`.

---

## Language Support

Reflex extracts static imports for the 15 supported code languages with Tree-sitter. Text, lock and generated files have no import extraction.

### Import/Dependency Syntax by Language

#### Rust
```rust
// Module imports
use std::collections::HashMap;           // stdlib
use crate::models::User;                 // internal (absolute)
use super::utils;                        // internal (relative)
use self::nested::Module;                // internal (self-relative)
mod my_module;                           // module declaration
extern crate serde;                      // external crate (Rust 2015)

// Tree-sitter patterns:
// - use_declaration
// - mod_item
// - extern_crate_declaration
```

#### Python
```python
import os                                 # stdlib
import requests                           # external
from typing import List, Optional         # stdlib
from .utils import format_date            # internal (relative)
from ..models import User                 # internal (parent)
from mypackage.module import func         # internal (absolute)

# Tree-sitter patterns:
# - import_statement
# - import_from_statement
```

#### JavaScript / TypeScript
```javascript
import React from 'react';                // external
import { useState } from 'react';         // external (named)
import * as utils from './utils';         // internal
import type { User } from './types';      // TS type import
const fs = require('fs');                 // CommonJS (stdlib)
const express = require('express');       // CommonJS (external)

// Tree-sitter patterns:
// - import_statement
// - import_clause
// - call_expression (for require)
```

#### Go
```go
import "fmt"                              // stdlib
import "github.com/user/repo/pkg"         // external
import . "math"                           // dot import
import _ "database/sql"                   // blank import
import (
    "context"
    "myapp/internal/models"               // internal
)

// Tree-sitter patterns:
// - import_declaration
// - import_spec
```

#### Java
```java
import java.util.List;                    // stdlib
import java.util.*;                       // wildcard
import com.example.myapp.User;            // internal
import org.springframework.boot.*;         // external

// Tree-sitter patterns:
// - import_declaration
```

#### C / C++
```c
#include <stdio.h>                        // stdlib (angle brackets)
#include "my_header.h"                    // internal (quotes)
#include <vector>                         // stdlib (C++)
#include "../common/utils.h"              // internal (relative)

// Tree-sitter patterns:
// - preproc_include
```

#### C#
```csharp
using System;                             // stdlib
using System.Collections.Generic;         // stdlib
using MyApp.Models;                       // internal
using static System.Math;                 // static using

// Tree-sitter patterns:
// - using_directive
```

#### PHP
```php
<?php
use PDO;                                  // stdlib
use Symfony\Component\HttpFoundation\Request;  // external
use App\Models\User;                      // internal
require_once 'config.php';                // include (internal)
include __DIR__ . '/utils.php';          // include (internal)

// Tree-sitter patterns:
// - namespace_use_declaration
// - require_expression
// - include_expression
```

#### Ruby
```ruby
require 'json'                            # stdlib
require 'rails'                           # external (gem)
require_relative '../lib/utils'          # internal (relative)
load 'config.rb'                          # load (internal)

# Tree-sitter patterns:
# - call (require/require_relative/load)
```

#### Kotlin
```kotlin
import java.util.Date                     // stdlib
import kotlin.collections.*               // stdlib (wildcard)
import com.example.myapp.User             // internal
import androidx.appcompat.app.AppCompatActivity  // external

// Tree-sitter patterns:
// - import_header
```

#### Zig
```zig
const std = @import("std");               // stdlib
const utils = @import("utils.zig");       // internal
const lib = @import("external_lib");      // external

// Tree-sitter patterns:
// - call_expression (@import)
```

#### Vue
```vue
<script>
import { ref } from 'vue'                 // external
import MyComponent from './MyComponent.vue'  // internal
import { api } from '@/services/api'      // internal (alias)
</script>

<script setup lang="ts">
import type { User } from '@/types'       // internal type import
</script>

// Parsing: Extract from <script> blocks using line-based parsing
```

#### Svelte
```svelte
<script>
import { onMount } from 'svelte'          // external
import Component from './Component.svelte'  // internal
import { store } from '../stores'         // internal
</script>

// Parsing: Extract from <script> blocks using line-based parsing
```

### Path Resolution

Every supported language has a resolver that maps an import to an indexed file
(`resolve_*_to_path` in `src/parsers/`). Project configuration is read where the language
needs it: `tsconfig.json` paths (TypeScript/JavaScript), `go.mod` (Go), `composer.json`
PSR-4 (PHP), Python package configs, Maven/Gradle layouts (Java/Kotlin), gemspecs (Ruby).
An import that resolves to no indexed file is `external` or `stdlib`.

Not implemented: npm/Cargo workspace members as separate packages, Python virtualenv
paths. C/C++ includes are resolved with `canonicalize()`, so `..` in a missing path does
not resolve (see `.context/TODO.md`).

---

## Use Cases & Examples

### Use Case 1: Safe Refactoring

**Scenario:** Rename `User` interface to `UserProfile`

**Without deps:**
```bash
# Agent searches and renames blindly
rfx query "interface User"
# Might miss indirect usages, breaks 30 files
```

**With deps:**
```bash
# Agent checks impact first
rfx query "interface User" --dependencies
# Sees imported by: auth.ts, api.ts, profile.ts, etc. (30 files)

rfx deps src/models/user.ts --reverse
# Gets complete list of all importers
# Makes informed decision about scope
```

### Use Case 2: Understanding Unfamiliar Code

**Scenario:** "How does authentication work?"

**Without deps:**
```bash
# Agent reads each file blindly
rfx query "auth" --symbols
# Finds 20 functions, reads all
```

**With deps:**
```bash
# Agent understands the stack immediately
rfx query "authenticateUser" --dependencies --json
{
  "symbol": "authenticateUser",
  "dependencies": [
    {"path": "jsonwebtoken", "type": "external"},
    {"path": "src/models/user.ts", "type": "internal"},
    {"path": "src/cache/redis.ts", "type": "internal"}
  ]
}
# "Ah, JWT tokens stored in Redis with User model"
```

### Use Case 3: Dead Code Detection

**Scenario:** "Can I delete this old module?"

```bash
rfx deps src/legacy/old-api.ts --reverse
# Returns: No files import this
# Safe to delete!
```

### Use Case 4: Architecture Audit

**Scenario:** Check for problematic patterns

```bash
# Find circular dependencies
rfx analyze --circular
→ Found 2 cycles:
  src/user.ts ↔ src/profile.ts
  src/api.ts → src/db.ts → src/models.ts → src/api.ts

# Find hotspots (files with too many dependents)
rfx analyze --hotspots --limit 10
→ src/config.ts (47 imports) ⚠️ God object
  src/utils/format.ts (33 imports)
  src/types.ts (28 imports)

# Find unused files
rfx analyze --unused
→ 5 orphaned files:
  src/old/unused.ts
  src/temp/test.ts
  ...
```

### Use Case 5: Debugging Import Errors

**Scenario:** Build failing due to import error

```bash
# Find what imports the problematic module
rfx query "UserService" --dependencies
→ Shows: imports './models/User'
→ But User.ts was renamed to UserModel.ts
→ "Found the issue!"
```

### Use Case 6: Find Usage Examples

**Scenario:** "How do I use the Logger class?"

```bash
# Find all files that import Logger
rfx deps src/logger.ts --reverse --limit 5
→ Returns 5 example files
→ Agent reads them to understand usage patterns
```

---


## Design history

The original 2025-11 specification, implementation plan and checklists were removed once
the feature shipped. Read them with `git show fc8da6b:docs/DEPENDENCIES.md`.
