# ADR-004: Clean-break domain ownership

Status: accepted.

The suite uses canonical domain packages and one `clumping` command tree. Old
module paths, console aliases, schema readers, organizers, and scheduler scripts
are not compatibility surfaces. Historical implementations and artifacts are
recoverable through Git history; keeping live shims would recreate multiple
owners and is therefore prohibited.
