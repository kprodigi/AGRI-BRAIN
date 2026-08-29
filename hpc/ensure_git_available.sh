#!/bin/bash
# Make Git available on Slurm workers before provenance validation.
#
# Some clusters expose Git on login nodes but only through an environment
# module on compute nodes.  Publication jobs must still fail closed when no
# verified checkout can be inspected, so this helper loads the cluster module
# only when Git is absent and then rechecks the executable explicitly.

if ! command -v git >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
        # The first candidate is the version currently advertised by the
        # SDSU/SDSMT cluster.  The generic name keeps the script portable to
        # module trees whose default Git version differs.
        for git_module in git/2.42.0 git; do
            if module load "$git_module" >/dev/null 2>&1; then
                break
            fi
        done
    fi
fi

if ! command -v git >/dev/null 2>&1; then
    echo "BLOCK: git executable is unavailable after module bootstrap."
    return 2
fi

