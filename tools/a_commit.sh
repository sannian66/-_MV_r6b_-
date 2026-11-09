#!/bin/bash
# A 组提交：只提交 A 组自己的路径。用法 tools/a_commit.sh "说明"
cd /workspace/work/mv
P=$(ls units/sec_a_intro_verse1.py units/a_*.py tools/a_*.py tools/a_commit.sh crew/round4/A_* out/round4/A_* assets/gen/A/* crew/genlog.md 2>/dev/null)
git add $P && GIT_AUTHOR_NAME=sannian66 GIT_AUTHOR_EMAIL=38859572+sannian66@users.noreply.github.com GIT_COMMITTER_NAME=sannian66 GIT_COMMITTER_EMAIL=38859572+sannian66@users.noreply.github.com git commit -q -m "[A组] $1" -- $P 2>&1 | grep -E "pushed|PUSH FAILED|error" 
