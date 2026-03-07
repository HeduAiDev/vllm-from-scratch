#!/bin/bash
# issue-manager.sh
# Issue管理工具

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

Issue管理工具

命令:
    list, ls              列出issues
    view, show            查看issue详情
    create, new           创建新issue
    comment, reply        回复issue
    close                 关闭issue
    reopen                重新打开issue
    label                 管理标签

选项:
    -n, --number NUM      Issue编号
    -s, --state STATE     状态: open, closed, all (默认: open)
    -l, --label LABEL     标签筛选
    -a, --assignee USER   指派人筛选
    -t, --title TITLE     Issue标题
    -b, --body BODY       Issue内容
    -c, --comment TEXT    评论内容
    -h, --help           显示帮助

示例:
    $0 list                           # 列出所有open issues
    $0 list -s all                    # 列出所有issues
    $0 list -l bug                    # 列出带bug标签的issues
    $0 view -n 123                    # 查看issue #123
    $0 create -t "Bug报告" -b "描述"   # 创建新issue
    $0 comment -n 123 -c "已修复"      # 回复issue
    $0 close -n 123                   # 关闭issue
    $0 label -n 123 -a bug            # 添加标签
    $0 label -n 123 -r bug            # 移除标签

EOF
    exit 1
}

# 列出issues
cmd_list() {
    local state="open"
    local label=""
    local assignee=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--state) state="$2"; shift 2 ;;
            -l|--label) label="$2"; shift 2 ;;
            -a|--assignee) assignee="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    local args=(--state "$state")
    [[ -n "$label" ]] && args+=(--label "$label")
    [[ -n "$assignee" ]] && args+=(--assignee "$assignee")
    
    print_header "=== Issues列表 ==="
    gh issue list "${args[@]}"
}

# 查看issue
cmd_view() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定issue编号: -n NUMBER"
        exit 1
    fi
    
    print_header "=== Issue #$number ==="
    gh issue view "$number"
    
    print_header "\n=== 评论 ==="
    gh issue view "$number" --comments
}

# 创建issue
cmd_create() {
    local title=""
    local body=""
    local labels=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--title) title="$2"; shift 2 ;;
            -b|--body) body="$2"; shift 2 ;;
            -l|--label) labels="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$title" ]]; then
        print_error "请指定标题: -t TITLE"
        exit 1
    fi
    
    local args=(--title "$title")
    [[ -n "$body" ]] && args+=(--body "$body")
    [[ -n "$labels" ]] && args+=(--label "$labels")
    
    print_info "创建issue..."
    gh issue create "${args[@]}"
}

# 评论issue
cmd_comment() {
    local number=""
    local comment=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            -c|--comment) comment="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定issue编号: -n NUMBER"
        exit 1
    fi
    
    if [[ -z "$comment" ]]; then
        print_error "请指定评论内容: -c TEXT"
        exit 1
    fi
    
    print_info "添加评论到issue #$number..."
    gh issue comment "$number" --body "$comment"
}

# 关闭issue
cmd_close() {
    local number=""
    local comment=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            -c|--comment) comment="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定issue编号: -n NUMBER"
        exit 1
    fi
    
    if [[ -n "$comment" ]]; then
        print_info "添加评论并关闭issue #$number..."
        gh issue close "$number" --comment "$comment"
    else
        print_info "关闭issue #$number..."
        gh issue close "$number"
    fi
}

# 重新打开issue
cmd_reopen() {
    local number=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定issue编号: -n NUMBER"
        exit 1
    fi
    
    print_info "重新打开issue #$number..."
    gh issue reopen "$number"
}

# 管理标签
cmd_label() {
    local number=""
    local add=""
    local remove=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--number) number="$2"; shift 2 ;;
            -a|--add) add="$2"; shift 2 ;;
            -r|--remove) remove="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$number" ]]; then
        print_error "请指定issue编号: -n NUMBER"
        exit 1
    fi
    
    if [[ -n "$add" ]]; then
        print_info "添加标签 '$add' 到issue #$number..."
        gh issue edit "$number" --add-label "$add"
    fi
    
    if [[ -n "$remove" ]]; then
        print_info "从issue #$number移除标签 '$remove'..."
        gh issue edit "$number" --remove-label "$remove"
    fi
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
    comment|reply)
        cmd_comment "$@"
        ;;
    close)
        cmd_close "$@"
        ;;
    reopen)
        cmd_reopen "$@"
        ;;
    label)
        cmd_label "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        print_error "未知命令: $COMMAND"
        usage
        ;;
esac
