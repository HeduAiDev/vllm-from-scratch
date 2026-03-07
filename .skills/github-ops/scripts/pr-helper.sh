#!/bin/bash
# pr-helper.sh
# Pull Request辅助工具

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

# 检查gh CLI
if ! command -v gh &> /dev/null; then
    print_error "请先安装GitHub CLI: https://cli.github.com/"
    exit 1
fi

# 显示用法
usage() {
    cat << EOF
用法: $0 [命令] [选项]

Pull Request辅助工具

命令:
    list, ls              列出PR
    view, show            查看PR详情
    create, new           创建PR
    checkout, co          检出PR到本地
    diff                  查看PR的diff
    review                审查PR
    merge                 合并PR
    checks                查看PR的checks状态

选项:
    -n, --number NUM      PR编号
    -s, --state STATE     状态: open, closed, merged, all (默认: open)
    -b, --base BRANCH     目标分支
    -t, --title TITLE     PR标题
    -d, --draft           创建为草稿PR
    -a, --approve         批准PR
    -r, --request-changes 请求修改
    -c, --comment TEXT    评论内容
    -m, --method METHOD   合并方法: merge, squash, rebase
    --delete-branch       合并后删除分支
    -h, --help           显示帮助

示例:
    $0 list                           # 列出所有open PR
    $0 list -s merged                 # 列出已合并的PR
    $0 view -n 456                    # 查看PR #456
    $0 checkout -n 456                # 检出PR到本地
    $0 diff -n 456                    # 查看PR的diff
    $0 create -b main -t "新功能"      # 创建PR
    $0 review -n 456 -a               # 批准PR
    $0 review -n 456 -r -c "需修改"    # 请求修改
    $0 merge -n 456                   # 合并PR
    $0 merge -n 456 -m squash         # Squash合并
    $0 checks -n 456                  # 查看checks状态

EOF
    exit 1
}

# 列出PR
cmd_list() {
    local state="open"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--state) state="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    print_header "=== Pull Requests ==="
    gh pr list --state "$state"
}

# 查看PR
cmd_view() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    print_header "=== PR #$number ==="
    gh pr view "$number"
}

# 创建PR
cmd_create() {
    local base=""
    local title=""
    local draft=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--base) base="$2"; shift 2 ;;
            -t|--title) title="$2"; shift 2 ;;
            -d|--draft) draft="--draft"; shift ;;
            *) shift ;;
        esac
    done
    
    local args=()
    [[ -n "$base" ]] && args+=(--base "$base")
    [[ -n "$title" ]] && args+=(--title "$title")
    [[ -n "$draft" ]] && args+=("$draft")
    
    print_info "创建Pull Request..."
    gh pr create "${args[@]}"
}

# 检出PR
cmd_checkout() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    print_info "检出PR #$number到本地..."
    gh pr checkout "$number"
}

# 查看diff
cmd_diff() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    print_header "=== PR #$number Diff ==="
    gh pr diff "$number"
}

# 审查PR
cmd_review() {
    local number=""
    local approve=""
    local request_changes=""
    local comment=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            -a|--approve) approve="true"; shift ;;
            -r|--request-changes) request_changes="true"; shift ;;
            -c|--comment) comment="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    if [[ "$approve" == "true" ]]; then
        print_info "批准PR #$number..."
        if [[ -n "$comment" ]]; then
            gh pr review "$number" --approve --body "$comment"
        else
            gh pr review "$number" --approve
        fi
    elif [[ "$request_changes" == "true" ]]; then
        print_info "请求修改PR #$number..."
        if [[ -n "$comment" ]]; then
            gh pr review "$number" --request-changes --body "$comment"
        else
            print_error "请求修改时必须提供评论: -c TEXT"
            exit 1
        fi
    else
        print_info "评论PR #$number..."
        if [[ -n "$comment" ]]; then
            gh pr review "$number" --comment --body "$comment"
        else
            gh pr review "$number" --comment
        fi
    fi
}

# 合并PR
cmd_merge() {
    local number=""
    local method="merge"
    local delete_branch=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            -m|--method) method="$2"; shift 2 ;;
            --delete-branch) delete_branch="--delete-branch"; shift ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    print_info "合并PR #$number (方法: $method)..."
    
    local args=(--"$method")
    [[ -n "$delete_branch" ]] && args+=("$delete_branch")
    
    gh pr merge "$number" "${args[@]}"
}

# 查看checks
cmd_checks() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定PR编号: -n NUMBER"
        exit 1
    fi
    
    print_header "=== PR #$number Checks ==="
    gh pr checks "$number"
}

# 主命令处理
COMMAND="${1:-}"
shift || true

case "$COMMAND" in
    list|ls)
        cmd_list "$@"
        ;;
    view|show)
        cmd_view "$@"
        ;;
    create|new)
        cmd_create "$@"
        ;;
    checkout|co)
        cmd_checkout "$@"
        ;;
    diff)
        cmd_diff "$@"
        ;;
    review)
        cmd_review "$@"
        ;;
    merge)
        cmd_merge "$@"
        ;;
    checks)
        cmd_checks "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        print_error "未知命令: $COMMAND"
        usage
        ;;
esac
