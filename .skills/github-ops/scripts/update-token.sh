#!/bin/bash
# update-token.sh
# 更新GitHub仓库的访问Token

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：显示用法
usage() {
    cat << EOF
用法: $0 [选项]

更新GitHub仓库的访问Token

选项:
    -t, --token TOKEN      新的GitHub Token (必需)
    -r, --repo REPO        仓库路径 (默认: 当前目录)
    -u, --url URL          完整的远程URL (可选)
    -h, --help            显示此帮助信息

示例:
    $0 -t ghp_xxxxxxxx
    $0 -t ghp_xxxxxxxx -r /path/to/repo
    $0 -t ghp_xxxxxxxx -u https://github.com/username/repo.git

EOF
    exit 1
}

# 解析参数
TOKEN=""
REPO_DIR="."
REMOTE_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--token)
            TOKEN="$2"
            shift 2
            ;;
        -r|--repo)
            REPO_DIR="$2"
            shift 2
            ;;
        -u|--url)
            REMOTE_URL="$2"
            shift 2
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

# 验证参数
if [[ -z "$TOKEN" ]]; then
    print_error "Token不能为空"
    usage
fi

# 进入仓库目录
cd "$REPO_DIR" || exit 1

# 检查是否是git仓库
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "当前目录不是git仓库: $REPO_DIR"
    exit 1
fi

print_info "当前仓库: $(pwd)"

# 获取当前远程URL
CURRENT_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [[ -z "$CURRENT_URL" ]]; then
    print_error "没有找到origin远程仓库"
    exit 1
fi

print_info "当前远程URL: $CURRENT_URL"

# 解析仓库信息
if [[ -z "$REMOTE_URL" ]]; then
    # 从当前URL提取仓库路径
    if [[ "$CURRENT_URL" =~ github.com[/:]([^/]+)/([^/]+)(\.git)?$ ]]; then
        USERNAME="${BASH_REMATCH[1]}"
        REPO="${BASH_REMATCH[2]}"
        REMOTE_URL="https://${TOKEN}@github.com/${USERNAME}/${REPO}.git"
    else
        print_error "无法解析当前远程URL"
        exit 1
    fi
fi

# 更新远程URL
print_info "正在更新远程URL..."
git remote set-url origin "$REMOTE_URL"

# 验证更新
NEW_URL=$(git remote get-url origin)
if [[ "$NEW_URL" == *"${TOKEN:0:10}"* ]]; then
    print_info "Token更新成功"
else
    print_error "Token更新失败"
    exit 1
fi

# 测试连接
print_info "测试连接..."
if git fetch origin --dry-run 2>/dev/null; then
    print_info "连接测试成功"
else
    print_warn "连接测试可能失败，请手动验证"
fi

print_info "完成！"
