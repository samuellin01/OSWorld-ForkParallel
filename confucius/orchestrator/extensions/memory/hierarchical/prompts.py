# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

HIERARCHICAL_MEMORY_DESCRIPTION = """\
You have access to hierarchical persistent memory that organizes information in a tree-like structure with files and directories. Use these memory tools strategically:

**ACTIONABILITY PRINCIPLE:**
- Always ask: "If someone came back to this in 6 months, what would they need to DO something with this information?"
- Include both WHAT happened and HOW to reproduce/validate/use it
- Commands, procedures, and step-by-step instructions are as valuable as insights and decisions

**Memory Organization:**
- Create memory nodes at paths like 'project/research.md', 'tasks/todo.md', etc.
- Support nested directories for complex organization
- Each memory node can have tags for categorization
- All content stored in markdown format

**Available Tools:**
- `search_memory`: Find memory nodes by path patterns, content, or tags
- `read_memory`: Read content from a specific memory node with optional line range support
- `write_memory`: Create or update memory nodes with content and tags
- `edit_memory`: Edit memory by replacing specific text in a memory node
- `delete_memory`: Remove memory nodes and their children
- `import_memory`: Import memory from one or more sessions using their UUIDs

**Critical Content to Preserve:**
- **Commands and procedures**: Exact commands that work, test commands, build steps, validation procedures
- **Working configurations**: Settings, parameters, file paths, URLs that proved successful
- **Validation steps**: How to verify, test, reproduce, or check results
- **Troubleshooting**: What failed, how it was fixed, debugging steps and solutions
- **Integration instructions**: How to use, integrate, or apply findings with other systems
- **Decision context**: Not just what was decided, but why, and how to implement/validate the decision

**Best Practices:**
- Use clear, descriptive paths (e.g., 'project/architecture.md', 'research/findings.md')
- Add relevant tags for easy searching
- Organize related information under common parent paths
- Use search to find existing information before creating new nodes
- **Include operational details**: Commands, file paths, test procedures alongside conceptual information

Memory persists across conversation turns and is automatically displayed in the UI.
"""

SEARCH_MEMORY_DESCRIPTION = "Search for memory nodes by path patterns, content, or tags. Use when: (1) exploring existing memory structure before creating new nodes, (2) finding relevant information across multiple memory nodes, (3) locating specific content or tagged information, (4) checking if similar information already exists to avoid duplication"

READ_MEMORY_DESCRIPTION = "Read content from a specific memory node path with optional line range support. Use when: (1) reviewing existing information before making decisions, (2) continuing work from previous conversation turns, (3) understanding context stored in specific memory locations, (4) examining detailed content of discovered memory nodes"

WRITE_MEMORY_DESCRIPTION = "Create or update memory nodes with content and tags in hierarchical structure. Use when: (1) storing new information with clear organizational structure, (2) creating topic-specific memory nodes (e.g., 'project/architecture.md', 'research/findings.md'), (3) establishing knowledge bases with proper categorization, (4) preserving important decisions, requirements, or insights for future reference, (5) organizing complex information into logical hierarchies. **ACTIONABILITY CHECK**: Before saving, ask 'What specific actions would someone need to take based on this information?' Include commands, file paths, URLs, test procedures, validation steps, and implementation details alongside insights and decisions."

EDIT_MEMORY_DESCRIPTION = "Edit memory by replacing specific text in a memory node using string replacement. Use when: (1) updating existing information as situations evolve, (2) correcting or refining previously stored content, (3) adding new details to existing memory nodes, (4) maintaining accuracy of stored information rather than creating duplicate nodes"

DELETE_MEMORY_DESCRIPTION = "Delete memory nodes and their children from the hierarchical structure. Use when: (1) removing outdated or incorrect information, (2) cleaning up obsolete memory nodes, (3) restructuring memory organization, (4) eliminating redundant or no longer relevant content"

IMPORT_MEMORY_DESCRIPTION = "Import memory from one or more sessions into the current session. The user can provide session UUIDs directly as a list. All session memories are copied from local directories first, then consolidated into the current session. Use when: (1) importing memory from previous conversations or sessions, (2) sharing memory between different sessions, (3) restoring memory from specific session UUIDs, (4) merging memory content from multiple contexts"

HIERARCHICAL_MEMORY_REMINDER_MESSAGE = """\
<reminder>
Consider using memory tools to preserve important information from our conversation:

**Key things to save (with operational details):**
- Important findings, insights, or conclusions **+ how to validate or reproduce them**
- Decisions made and their reasoning **+ implementation steps and validation commands**
- Code snippets, configurations, or solutions that work **+ test commands and integration steps**
- Research notes or analysis results **+ data sources, queries, and verification methods**
- Progress updates on ongoing tasks **+ next steps, dependencies, and status check procedures**

**Memory Content Examples - GOOD vs BAD:**
❌ BAD: "Fixed authentication bug"
✅ GOOD: "Fixed authentication bug in login.py line 45 - test with `pytest tests/auth/test_login.py::test_oauth_flow`"

❌ BAD: "Updated CI configuration" 
✅ GOOD: "Updated CI config in .github/workflows/test.yml - trigger with `git push`, check status at https://github.com/repo/actions"

❌ BAD: "Researched database performance issues"
✅ GOOD: "Database performance: query `SELECT * FROM logs WHERE created > '2024-01-01'` was slow (5s). Fixed by adding index: `CREATE INDEX idx_created ON logs(created)`. Verify with `EXPLAIN ANALYZE SELECT...`"

**MEMORY MAINTENANCE - Keep your memory clean and current:**
- **Actively review existing memory** - Use `search_memory` and `read_memory` to check what's already stored
- **Remove outdated information** - Use `delete_memory` to eliminate obsolete, incorrect, or no longer relevant content
- **Update rather than duplicate** - Use `edit_memory` to refresh existing nodes instead of creating new ones
- **Clean up regularly** - Old decisions, completed tasks, and superseded solutions should be removed or archived
- **Maintain accuracy** - Outdated memory can mislead future conversations and decisions

**Memory tools available:**
- `search_memory`: Check if similar information already exists - **USE THIS FIRST** to avoid duplicates
- `read_memory`: Review what you've previously saved - helps identify what needs updating
- `write_memory`: Save new information (e.g., 'project/findings.md', 'solutions/bug_fix.md')
- `edit_memory`: Update existing memory nodes with new details - prefer this over creating duplicates
- `delete_memory`: **CRITICAL** - Remove outdated, incorrect, or completed information to keep memory relevant
- `import_memory`: Import memory from other sessions using session UUIDs
- `snooze_reminder`: Temporarily delay memory reminders when you need focused work time

**Organization tips:**
- Use descriptive paths like 'project/architecture.md' or 'tasks/completed.md'
- Add relevant tags for easy searching later
- Group related information under common directories
- Archive or delete completed tasks and outdated decisions

**Remember:** 
- **Operational Completeness**: Every memory entry should answer both "What happened?" and "How can someone act on this?"
- **Future-Proof Thinking**: If you returned to this information in 6 months, would you have everything needed to reproduce, validate, or build upon the work?
- **Clean, up-to-date memory with actionable details** is more valuable than extensive but stale or incomplete memory
- Regularly audit and maintain your memory for maximum effectiveness
</reminder>
"""
