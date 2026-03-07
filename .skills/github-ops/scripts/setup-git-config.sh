#!/bin/bash
# setup-git-config.sh
# 配置git提交规范和其他常用设置

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

usage() {
    cat << EOF
用法: $0 [选项]

配置git提交规范和其他常用设置

选项:
    -n, --name NAME       设置git用户名
    -e, --email EMAIL     设置git邮箱
    -g, --global          使用全局配置（默认）
    -l, --local           使用本地配置
    -h, --help           显示帮助

示例:
    $0 -n "张三" -e "zhangsan@example.com"
    $0 -n "张三" -e "zhangsan@example.com" -l

EOF
    exit 1
}

SCOPE="--global"
NAME=""
EMAIL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            NAME="$2"
            shift 2
            ;;
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -g|--global)
            SCOPE="--global"
            shift
            ;;
        -l|--local)
            SCOPE="--local"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "未知选项: $1"
            usage
            ;;
    esac
done

print_header "=== 配置Git ==="

# 设置用户名
if [[ -n "$NAME" ]]; then
    print_info "设置用户名为: $NAME"
    git config $SCOPE user.name "$NAME"
fi

# 设置邮箱
if [[ -n "$EMAIL" ]]; then
    print_info "设置邮箱为: $EMAIL"
    git config $SCOPE user.email "$EMAIL"
fi

# 设置默认分支名
print_info "设置默认分支名为 main"
git config $SCOPE init.defaultBranch main

# 设置pull策略为rebase（避免merge commit）
print_info "设置pull策略为rebase"
git config $SCOPE pull.rebase true

# 设置push策略为simple
git config $SCOPE push.default simple

# 设置编辑器
if command -v vim &> /dev/null; then
    print_info "设置编辑器为vim"
    git config $SCOPE core.editor vim
elif command -v nano &> /dev/null; then
    print_info "设置编辑器为nano"
    git config $SCOPE core.editor nano
fi

# 设置颜色输出
git config $SCOPE color.ui auto

# 设置别名
print_info "设置常用别名"
git config $SCOPE alias.st status
git config $SCOPE alias.co checkout
git config $SCOPE alias.br branch
git config $SCOPE alias.ci commit
git config $SCOPE alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
git config $SCOPE alias.last "log -1 HEAD"
git config $SCOPE alias.unstage "reset HEAD --"
git config $SCOPE alias.visual "!gitk"

print_header "=== Git配置完成 ==="
echo ""
git config $SCOPE --list | grep -E "(user|core|alias|init|pull|push)"
