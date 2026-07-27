#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_ROOT="/var/www/blog"

echo "📦 构建项目..."
cd "$PROJECT_DIR"
npm run build

echo "📁 部署到 $WEB_ROOT ..."
mkdir -p "$WEB_ROOT"
rsync -avz --delete "$PROJECT_DIR/dist/" "$WEB_ROOT/"

echo "🔄 重载 Nginx..."
systemctl reload nginx

echo "✅ 部署完成！访问 https://blog.rzhang-lab.cn"
