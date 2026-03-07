# GitHub 仓库运营技能文档

本技能文档总结了GitHub仓库运营的核心经验，包括Token管理、Issue管理、PR管理和最佳实践。

---

## 1. GitHub Token 管理

### 1.1 什么是GitHub Token

GitHub Token（个人访问令牌）是用于替代密码进行Git操作的认证方式。相比密码，Token具有以下优势：
- 可以设置细粒度的权限范围
- 可以设置过期时间
- 可以随时撤销
- 支持多因素认证

### 1.2 如何配置git使用Token

#### 方法一：直接嵌入URL（推荐用于自动化）

```bash
# 设置远程仓库URL，包含Token
git remote set-url origin https://TOKEN@github.com/username/repo.git

# 示例
git remote set-url origin https://ghp_xxxxxxxx@github.com/HeduAiDev/vllm-from-scratch.git
```

#### 方法二：使用Git凭证助手

```bash
# 配置凭证存储
git config --global credential.helper store

# 第一次推送时输入用户名和Token，之后自动保存
git push origin main
```

#### 方法三：使用环境变量

```bash
# 设置环境变量
export GITHUB_TOKEN=ghp_xxxxxxxx

# 在.gitconfig中使用变量
# [url "https://${GITHUB_TOKEN}@github.com/"]
#     insteadOf = https://github.com/
```

### 1.3 Token的安全存储方法

1. **不要硬编码Token在代码中**
   - 使用环境变量
   - 使用配置文件（加入.gitignore）
   - 使用密钥管理服务

2. **使用Git凭证管理器**
   ```bash
   # macOS
   git config --global credential.helper osxkeychain
   
   # Windows
   git config --global credential.helper manager
   
   # Linux
   git config --global credential.helper libsecret
   ```

3. **定期轮换Token**
   - 设置合理的过期时间
   - 定期生成新Token并更新配置

### 1.4 Token过期/失效的处理

当Token过期或失效时，会出现以下错误：
```
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/...'
```

处理步骤：
1. 登录GitHub生成新Token
2. 更新本地git配置：
   ```bash
   # 查看当前远程URL
   git remote -v
   
   # 更新为新Token
   git remote set-url origin https://NEW_TOKEN@github.com/username/repo.git
   ```
3. 如果使用凭证助手，清除旧凭证：
   ```bash
   # 清除特定主机的凭证
   git credential reject
   protocol=https
   host=github.com
   
   # 重新推送，输入新Token
   git push
   ```

---

## 2. Issue 管理

### 2.1 查看仓库的Issues

#### 命令行方式（使用gh CLI）

```bash
# 安装GitHub CLI
# macOS: brew install gh
# Windows: winget install --id GitHub.cli
# Ubuntu: sudo apt install gh

# 登录
gh auth login

# 列出所有open的issues
gh issue list

# 列出所有issues（包括closed）
gh issue list --state all

# 按标签筛选
gh issue list --label bug

# 按作者筛选
gh issue list --author username

# 查看特定issue详情
gh issue view 123
```

#### Web界面
直接访问：`https://github.com/username/repo/issues`

### 2.2 回复Issue

```bash
# 在命令行回复
ghtoken issue comment 123 --body "感谢反馈，我会尽快处理"

# 使用文件内容回复
ghtoken issue comment 123 --body-file response.md
```

### 2.3 关闭Issue

```bash
# 关闭issue
ghtoken issue close 123

# 关闭并评论
ghtoken issue close 123 --comment "问题已修复"

# 重新打开
ghtoken issue reopen 123
```

### 2.4 Issue标签管理

```bash
# 列出所有标签
ghtoken label list

# 给issue添加标签
ghtoken issue edit 123 --add-label bug,critical

# 移除标签
ghtoken issue edit 123 --remove-label bug

# 创建新标签
ghtoken label create "help wanted" --color "008672" --description "需要帮助"
```

**常用标签建议：**
- `bug` - 缺陷
- `enhancement` - 功能增强
- `documentation` - 文档相关
- `good first issue` - 适合新手
- `help wanted` - 需要帮助
- `wontfix` - 不修复
- `duplicate` - 重复

---

## 3. Pull Request 管理

### 3.1 查看PR

```bash
# 列出所有open的PR
gh pr list

# 列出所有PR
gh pr list --state all

# 按作者筛选
gh pr list --author username

# 查看特定PR详情
gh pr view 456

# 在浏览器中打开
gh pr view 456 --web
```

### 3.2 审查PR

```bash
# 检出PR到本地分支
gh pr checkout 456

# 查看PR的diff
gh pr diff 456

# 查看PR的checks状态
gh pr checks 456

# 审查PR
ghtoken pr review 456 --approve --body "LGTM!"
ghtoken pr review 456 --request-changes --body "需要修改..."
ghtoken pr review 456 --comment --body "有一个小问题..."
```

### 3.3 合并PR

```bash
# 合并PR（创建merge commit）
ghtoken pr merge 456

# Squash合并
ghtoken pr merge 456 --squash

# Rebase合并
ghtoken pr merge 456 --rebase

# 合并并删除分支
ghtoken pr merge 456 --delete-branch

# 自动合并（当checks通过时）
ghtoken pr merge 456 --auto
```

---

## 4. 仓库维护最佳实践

### 4.1 提交信息规范

#### 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type类型
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具相关

#### 示例
```
feat(model): 添加GPT-2模型支持

- 实现GPT-2的tokenizer
- 添加模型配置
- 更新文档

Closes #123
```

#### 工具推荐
```bash
# 使用commitizen规范提交
npm install -g commitizen
git cz  # 替代git commit
```

### 4.2 分支管理策略

#### Git Flow（适合大型项目）
```
main/master: 生产分支
  ↓
develop: 开发分支
  ↓
feature/*: 功能分支
release/*: 发布分支
hotfix/*: 热修复分支
```

#### GitHub Flow（适合敏捷开发）
```
main: 稳定分支
  ↓
feature-branch: 功能分支 → PR → 合并到main
```

#### 分支命名规范
```
feature/add-login-page
bugfix/fix-memory-leak
hotfix/security-patch
release/v1.2.0
```

#### 保护分支设置
在GitHub Settings > Branches中设置：
- 要求PR审查
- 要求状态检查通过
- 禁止强制推送
- 禁止直接推送

### 4.3 文档维护

#### README.md结构
```markdown
# 项目名

## 简介
简短描述项目是什么，解决什么问题

## 功能特性
- 特性1
- 特性2

## 安装
```bash
pip install xxx
```

## 快速开始
最小可运行的示例

## 文档
链接到完整文档

## 贡献指南
如何参与贡献

## 许可证
MIT/Apache等
```

#### 文档工具
- **Wiki**: 适合详细文档
- **GitHub Pages**: 适合项目网站
- **ReadTheDocs**: 适合大型文档项目

#### 自动化文档
```bash
# 使用GitHub Actions自动生成文档
git add .github/workflows/docs.yml
```

---

## 5. 常用脚本

见 `scripts/` 目录下的实用脚本：
- `update-token.sh` - 更新GitHub Token
- `issue-manager.sh` - Issue批量管理
- `pr-helper.sh` - PR辅助工具

---

## 6. 参考资源

- [GitHub CLI文档](https://cli.github.com/manual/)
- [GitHub Docs](https://docs.github.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
