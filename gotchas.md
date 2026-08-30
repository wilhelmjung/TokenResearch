# 项目开发与文档发布避坑指南 (Gotchas & Best Practices)

> **定位**：本文件系统归档在撰写技术报告、生成 HTML 演示大屏、部署 GitHub Pages 以及处理数学公式与特殊符号时的**常见格式陷阱（Gotchas）与标准化防范规则**，避免在后续迭代中重复出现格式与渲染异常。

---

## 📌 格式与渲染陷阱清单 (Formatting Gotchas)

### 1. HTML 页面中严禁遗留未渲染的裸 LaTeX 语法
* **问题现象**：
  * 在编写 HTML / 演示文稿时，直接复制粘贴 Markdown 中的 LaTeX 语法，如 `$\rightarrow$` 或 `$\min \sum I_{ij}(W - \hat{W})^2$`；
  * 若页面未引入 MathJax / KaTeX 运行时，浏览器将**原样输出原始反斜杠与花括号字符**（如 `$\rightarrow$`、`\hat{W}`），严重破坏大屏与报告美观度。
* **防范规则**：
  1. **纯静态 HTML 优先方案**：一律使用 **Unicode 符号 + HTML 语义标签**：
     * 箭头：使用 `➔` 或 `→`；
     * 乘号：使用 `×` 或 `·`；
     * 上下标：使用 `<sub>ij</sub>`、`<sup>2</sup>`；
     * 公式代码块：使用 `<code class="font-mono ...">min Σ I<sub>ij</sub>·(W - Ŵ)²</code>`；
  2. **复杂公式渲染方案**：如确需渲染复杂多行 LaTeX，必须在 `<head>` 中显式引入 KaTeX 脚本与 `renderMathInElement` 自动渲染器。

---

### 2. Markdown 中的美元符号（`$`）价格转义陷阱
* **问题现象**：
  * 在 Markdown 或 HTML 文本中连续提及价格（例如 `$0.22` 和 `$0.007`）；
  * 两个相邻的 `$` 会被 Markdown 解析器（如 KaTeX / GitHub Markdown）**误当成 LaTeX 行内公式起止符**，导致两个价格之间的整段文字全变成数学斜体，甚至发生语法解析中断。
* **防范规则**：
  * 涉及到美元价格时，一律使用转义符 `\$0.22`，或使用行内反引号包裹：`$0.22/M`。

---

### 3. GitHub Pages 默认 Jekyll 构建过滤陷阱
* **问题现象**：
  * GitHub Pages 默认启用 Jekyll 引擎，Jekyll 会自动忽略以下划线（`_`）或点（`.`）开头的目录（例如 `.agents/`、`_agents/`、`_static/`），导致静态资源 404；
  * 页面若包含类似 `{{ variable }}` 的文本，会被 Jekyll 当成 Liquid 模版错误拦截并导致构建失败（`Page build failed`）。
* **防范规则**：
  * 在仓库根目录必须放置空的 **`.nojekyll`** 文件，强制 GitHub Pages 关闭 Jekyll，直接作为纯静态 Web 服务器分发。

---

### 4. 跨平台编码与中英文混排空格规范
* **问题现象**：
  * 中英文、数字与专有名词紧挨无间距时（例如 `DeepSeek671B模型`），在部分移动端字体渲染下会产生字偶距粘连。
* **防范规则**：
  * 中文与英文字符/数字之间保持一个半角空格（例如 `DeepSeek 671B 模型`、`8 维向量码本`）；
  * 专有名词保持官方统一定义（如 `Jalapeño`、`DFlash`、`vLLM`、`SGLang`、`DS4.c`）。

---

### 5. 跨文档编号连续性与主文档引用契约
* **规则要求（遵循 `AGENTS.md` 契约）**：
  * 序号 1 的文档永远为主文档（`Topic-1-TokenEconomy/1-Infra-TokenCostReduction-Report.md`）；
  * 后续新增独立技术专题必须按严格连续数字命名（`2-...` 至 `10-...`），并在主文档、`README.md`、`Report/` 索引及在线大屏中完成同步双向引用。
