dir="absolute_bert"

for file in "$dir"/*; do
  if [ -f "$file" ]; then
    echo "📄 $file"

    # 查新增 commit（A）
    git log --follow --diff-filter=A --format="  [ADD] %ad (%h)" --date=short -- "$file" | tail -n 1

    # 查 rename commit（R）
    git log --follow --diff-filter=R --format="  [RENAME] %ad (%h)" --date=short -- "$file" | tail -n 1
    echo ""
  fi
done
