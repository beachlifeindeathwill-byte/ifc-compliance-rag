import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowRight, ArrowsClockwise, ArrowsLeftRight, Buildings, CaretDown, ChatCircleDots, CheckCircle, ClockCounterClockwise,
  Database, DownloadSimple, FileArrowUp, FileText, Fire, Gear, GitBranch, MagnifyingGlass,
  DiceFive, PaperPlaneRight, PencilSimple, Plus, ShieldCheck, SidebarSimple, Sparkle, SpinnerGap,
  Trash, UploadSimple, WarningCircle, X,
} from '@phosphor-icons/react'
import { api } from './api'

const modules = [
  { id: 'qa', label: '规范问答', icon: MagnifyingGlass },
  { id: 'bim', label: 'BIM 模型审查', icon: Buildings },
  { id: 'versions', label: '版本对比', icon: GitBranch },
  { id: 'library', label: '规范库', icon: Database },
  { id: 'history', label: '查询历史', icon: ClockCounterClockwise },
  { id: 'status', label: '系统状态', icon: Gear },
]

const examples = [
  '商业综合体中庭四周的防火卷帘设置有什么具体要求？',
  '汽车库室内任一点到最近人员安全出口的疏散距离是多少？',
  '高层公共建筑疏散楼梯的最小净宽是多少？',
]

const confidenceLabel = (level) => ({ high: '高', medium: '中', low: '低' })[level] || '待判断'

function readStoredJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || '') || fallback } catch { return fallback }
}

function writeStoredJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
    return true
  } catch {
    return false
  }
}

function makeSession(title = '新会话') {
  const now = new Date().toISOString()
  return {
    id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title,
    createdAt: now,
    updatedAt: now,
    qaTurns: [],
    projectContext: '',
    ifcData: null,
    ifcFileName: '',
    ifcFileSize: 0,
    ifcModels: [],
    ifcActiveModelId: '',
    projectDate: '',
    modelAnswers: [],
    reviewAnswers: [],
    projectProfile: null,
    projectProfileContext: '',
    reviewQuestions: [],
    reviewPreparationStatus: '',
  }
}

function formatSessionTime(value) {
  if (!value) return ''
  const date = new Date(value)
  return `${date.getMonth() + 1}月${date.getDate()}日 ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

function SessionPanel({ sessions, activeSessionId, onSelect, onCreate, onRename, onDelete }) {
  return <aside className="session-panel">
    <div className="session-panel-head">
      <h2>历史问答</h2>
      <button className="icon-button session-new" onClick={onCreate} title="新建会话"><Plus size={18} /></button>
    </div>
    <div className="session-list">
      {sessions.map((session) => <div className={`session-item ${session.id === activeSessionId ? 'active' : ''}`} key={session.id}>
        <button className="session-select" onClick={() => onSelect(session.id)} type="button">
          <span className="session-dot" />
          <span className="session-copy"><strong>{session.title}</strong><small>{formatSessionTime(session.updatedAt)}</small></span>
        </button>
        <div className="session-actions">
          <button onClick={() => onRename(session.id)} title="重命名" type="button"><PencilSimple size={13} /></button>
          <button onClick={() => onDelete(session.id)} title="删除会话" type="button"><Trash size={13} /></button>
        </div>
      </div>)}
    </div>
  </aside>
}

const annotationStopWords = new Set([
  '多少', '什么', '是否', '怎么', '怎样', '要求', '可以', '需要', '如果',
  '按照', '这个', '那个', '哪些', '一下', '请问', '说明', '判断',
  '按', '请', '的', '了', '是', '吗', '呢', '么',
])

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

function cleanDisplayText(value) {
  return String(value || '')
    .replace(/\r/g, '')
    .replace(/[ \t]+/g, ' ')
    .replace(/([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])/g, '$1')
    .replace(/([\u4e00-\u9fff])\s+([，。；：、？！])/g, '$1$2')
    .replace(/([，。；：、？！])\s+([\u4e00-\u9fff])/g, '$1$2')
    .replace(/\s+([;:])/g, '$1')
    .replace(/([，。；：、？！;:])\s+(?=[0-9A-Za-z])/g, '$1')
    .replace(/([，。；：、？！;:])\s+([\u4e00-\u9fff])/g, '$1$2')
    .replace(/([\u4e00-\u9fff])\s+(?=[0-9A-Za-z])/g, '$1')
    .replace(/([0-9A-Za-z])\s+([\u4e00-\u9fff])/g, '$1$2')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

function downloadTextFile(filename, text, type = 'text/markdown;charset=utf-8') {
  const blob = new Blob([text], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function questionKeywords(question = '') {
  if (!question.trim() || typeof Intl.Segmenter !== 'function') return []
  const segmenter = new Intl.Segmenter('zh-CN', { granularity: 'word' })
  const phrases = question.split(/[，。；！？,.!?;：:]/).flatMap((clause) => {
    const words = [...segmenter.segment(clause)]
      .filter((item) => item.isWordLike)
      .map((item) => item.segment.trim())
      .filter((item) => item && !annotationStopWords.has(item) && !/^\d+$/.test(item))
    const clausePhrases = []
    for (let size = 4; size >= 2; size -= 1) {
      for (let index = 0; index <= words.length - size; index += 1) {
        const phrase = words.slice(index, index + size).join('')
        if (phrase.length >= 4 && phrase.length <= 18) clausePhrases.push(phrase)
      }
    }
    return clausePhrases
  })
  return phrases
    .filter((item, index, items) => items.indexOf(item) === index)
    .sort((a, b) => b.length - a.length)
    .slice(0, 24)
}

function annotationKind(value) {
  if (/^GB\s*\d{5}/i.test(value)) return 'standard'
  if (/^第\s*\d+(?:\.\d+)+\s*条|^PDF\s*物理页/i.test(value)) return 'reference'
  if (/\d+(?:\.\d+)?\s*(?:m²|㎡|m|米|h|小时|%|人|辆|层)$/i.test(value)) return 'value'
  return 'subject'
}

function AnnotatedText({ text, question }) {
  if (!text) return null
  const structuralPatterns = [
    'GB\\s*\\d{5}(?:-\\d{4})?',
    '第\\s*\\d+(?:\\.\\d+){1,3}\\s*条(?:第\\s*\\d+\\s*款)?',
    'PDF\\s*物理页第\\s*\\d+\\s*页',
    '\\d+(?:\\.\\d+)?\\s*(?:m²|㎡|m|米|h|小时|%|人|辆|层)',
  ]
  const patterns = [...structuralPatterns, ...questionKeywords(question).map(escapeRegExp)]
  const matcher = new RegExp(`(${patterns.join('|')})`, 'gi')
  return cleanDisplayText(text).split(matcher).map((part, index) => {
    if (!part || !new RegExp(`^(?:${patterns.join('|')})$`, 'i').test(part)) return part
    return <mark className={`annotation ${annotationKind(part)}`} key={`${part}-${index}`}>{part}</mark>
  })
}

const versionHighlightPatterns = [
  'GB\\s*\\d{5}(?:-\\d{4})?',
  '\\d{4}-\\d{2}-\\d{2}',
  '第\\s*\\d+(?:\\.\\d+){1,3}\\s*条(?:第\\s*\\d+\\s*款)?',
  '\\d+(?:\\.\\d+)?\\s*(?:m²|㎡|m|米|h|小时|%|人|辆|层)',
  '(?:不应小于|不应大于|新建|既有|自动灭火系统|替代|废止|优先核验)',
]

function versionHighlightKind(part) {
  if (/^GB\s*\d{5}/i.test(part)) return 'standard'
  if (/^\d{4}-\d{2}-\d{2}$/.test(part)) return 'reference'
  if (/^第\s*\d|^PDF/i.test(part)) return 'reference'
  if (/\d/.test(part)) return 'value'
  return 'subject'
}

function VersionHighlight({ text }) {
  if (!text) return null
  const matcher = new RegExp(`(${versionHighlightPatterns.join('|')})`, 'gi')
  return cleanDisplayText(text).split(matcher).map((part, index) => {
    if (!part || !new RegExp(`^(?:${versionHighlightPatterns.join('|')})$`, 'i').test(part)) return part
    return <mark className={`annotation ${versionHighlightKind(part)}`} key={`${part}-${index}`}>{part}</mark>
  })
}

function AppShell({ active, setActive, library, health, expert, setExpert, sessions, activeSessionId, onSelectSession, onCreateSession, onRenameSession, onDeleteSession, storageWarning, children }) {
  const [collapsed, setCollapsed] = useState(false)
  const activeItem = modules.find((item) => item.id === active)
  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-head">
          <div className="product-mark"><Fire size={19} weight="fill" /></div>
          {!collapsed && <span>消防审查</span>}
          <button className="icon-button sidebar-toggle" onClick={() => setCollapsed(!collapsed)} title={collapsed ? '展开导航' : '收起导航'}><SidebarSimple size={18} /></button>
        </div>
        <nav className="nav-list" aria-label="主导航">
          {modules.map((item) => {
            const Icon = item.icon
            return <button key={item.id} className={`nav-item ${active === item.id ? 'active' : ''}`} onClick={() => setActive(item.id)} title={item.label}><Icon size={19} weight={active === item.id ? 'fill' : 'regular'} />{!collapsed && <span>{item.label}</span>}</button>
          })}
        </nav>
        {(active === 'qa' || active === 'bim') && !collapsed && <SessionPanel sessions={sessions} activeSessionId={activeSessionId} onSelect={onSelectSession} onCreate={onCreateSession} onRename={onRenameSession} onDelete={onDeleteSession} />}
        <div className="sidebar-bottom">
          {!collapsed && <div className="library-mini"><span className="mini-label">当前知识库</span><strong>{library?.standards?.length || 0} 份规范</strong><span>{library?.total_chunks || 0} 个可检索片段</span></div>}
          <div className="service-line"><span className={`status-dot ${health?.status === 'ok' ? 'online' : ''}`} />{!collapsed && <span>{health?.status === 'ok' ? '服务正常' : '正在连接'}</span>}</div>
        </div>
      </aside>
      <section className="app-stage">
        <header className="topbar">
          <div><h1>{activeItem?.label}</h1><p>{active === 'qa' ? '独立检索规范并生成可核查回答' : active === 'bim' ? '解析模型事实，分别进行模型查询与合规审查' : active === 'versions' ? '按施行时间对比规范条文变化' : '消防规范知识与系统支持'}</p></div>
          <label className="expert-switch"><input type="checkbox" checked={expert} onChange={(event) => setExpert(event.target.checked)} /><span>专家模式</span></label>
        </header>
        {storageWarning && <div className="storage-warning"><WarningCircle size={15} /><span>{storageWarning}</span></div>}
        <main className="workspace">{children}</main>
      </section>
    </div>
  )
}

function ScopeSelect({ library, value, onChange }) {
  return <label className="scope-select"><span>知识库范围</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="auto">自动识别</option><option value="all">全部规范</option>{(library?.standards || []).filter((item) => item.standard_id !== '未配置').map((item) => <option value={item.standard_id} key={`${item.standard_id}-${item.source_file}`}>{item.source_file}</option>)}</select></label>
}

function CitationCountControl({ value, onChange }) {
  return <label className="citation-count"><span>引用依据数量</span><input type="range" min="3" max="10" step="1" value={value} onChange={(event) => onChange(Number(event.target.value))} /><strong>{value} 条</strong></label>
}

function SourceList({ sources, open, onToggle, expert, question }) {
  return <div className="sources-block">
    <button className="disclosure" onClick={onToggle} aria-expanded={open}><span><FileText size={17} />查看引用来源（{sources.length} 条）</span><CaretDown size={16} className={open ? 'rotated' : ''} /></button>
    {open && <div className="source-list">{sources.map((source, index) => <article className="source-row" key={source.chunk_id || `${source.source_file}-${index}`}><div className="source-index">{index + 1}</div><div className="source-content"><div className="source-meta"><strong>{source.source_file}</strong><span>{source.standard_id}</span><span>第 {source.article_no} 条</span><span>{source.page == null ? '页码未识别' : `PDF 物理页第 ${source.page} 页`}</span></div><p><AnnotatedText text={source.text} question={question} /></p>{expert && <div className="retrieval-meta">排序分数 {source.score}{source.notes?.length ? ` · ${source.notes.join(' · ')}` : ''}</div>}</div></article>)}</div>}
  </div>
}

function AnswerCard({ turn, expert }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [confidenceOpen, setConfidenceOpen] = useState(false)
  const answer = turn.answer || {}
  const evidence = turn.evidence || {}
  return <div className="turn">
    <div className="user-message"><span>提问</span><p>{turn.question}</p></div>
    {turn.context?.used && <div className="context-note"><Sparkle size={15} />已结合上一轮上下文理解本次追问</div>}
    <article className="answer-card"><div className="answer-rail" /><div className="answer-body">
      <div className="answer-heading"><ShieldCheck size={19} weight="fill" /><span>规范顾问</span></div>
      <p className="conclusion"><AnnotatedText text={answer.conclusion || '未生成回答'} question={turn.question} /></p>
      {answer.basis && <p className="basis"><span>依据</span><AnnotatedText text={answer.basis} question={turn.question} /></p>}
      <div className="confidence-row"><span className={`confidence ${evidence.level || 'low'}`}>证据置信度：{confidenceLabel(evidence.level)}</span><span className={`confidence ${answer.certainty || 'low'}`}>模型判断：{confidenceLabel(answer.certainty)}</span></div>
      {answer.missing_fields?.length > 0 && <div className="missing-fields"><WarningCircle size={16} /><span>仍需确认：{answer.missing_fields.join('、')}</span></div>}
      <button className="confidence-toggle" onClick={() => setConfidenceOpen(!confidenceOpen)}>置信度与复核说明 <CaretDown size={14} className={confidenceOpen ? 'rotated' : ''} /></button>
      {confidenceOpen && <div className="confidence-detail"><p><strong>证据判断：</strong>{evidence.reason || '暂无说明'}</p><p><strong>模型判断：</strong>{answer.confidence_reason || answer.note || '暂无说明'}</p>{expert && turn.context && <p><strong>上下文路由：</strong>{turn.context.reason}</p>}</div>}
      <SourceList sources={turn.sources || []} open={sourcesOpen} onToggle={() => setSourcesOpen(!sourcesOpen)} expert={expert} question={turn.question} />
    </div></article>
  </div>
}

function Composer({ value, setValue, onSend, loading, placeholder }) {
  const submit = (event) => { event.preventDefault(); if (value.trim() && !loading) onSend(value.trim()) }
  return <form className="composer" onSubmit={submit}><input value={value} onChange={(event) => setValue(event.target.value)} placeholder={placeholder} /><button type="submit" disabled={!value.trim() || loading} title="发送">{loading ? <SpinnerGap size={19} className="spin" /> : <PaperPlaneRight size={19} weight="fill" />}</button></form>
}

function GenerationControl({ loading, stopped, onStop }) {
  if (loading) {
    return <div className="loading-line"><SpinnerGap size={18} className="spin" />正在检索并生成回答...<button className="stop-button" onClick={onStop} type="button"><X size={15} />停止生成</button></div>
  }
  if (stopped) {
    return <div className="stopped-line">已停止生成，本轮未写入对话。</div>
  }
  return null
}

function SessionSwitch({ sessions, activeSessionId, onSelect, onCreate, onClear }) {
  return <div className="session-switch" role="group" aria-label="会话操作">
    <label className="session-picker"><span>会话</span>
      <select value={activeSessionId} onChange={(event) => onSelect(event.target.value)} aria-label="切换会话">
        {sessions.map((session) => <option key={session.id} value={session.id}>{session.title}</option>)}
      </select>
    </label>
    <button className="icon-button" onClick={onCreate} title="新建会话" type="button"><Plus size={15} /></button>
    <button className="text-button review-clear" onClick={onClear} title="清空当前审查对话" type="button"><Trash size={14} />清空对话</button>
  </div>
}

function QAWorkspace({ library, expert, onHistory, session, onUpdateSession }) {
  const [scope, setScope] = useState('auto')
  const [topK, setTopK] = useState(5)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [stopped, setStopped] = useState(false)
  const [error, setError] = useState('')
  const scrollRef = useRef(null)
  const abortRef = useRef(null)
  const turns = session.qaTurns || []
  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' }) }, [turns, loading])
  const send = async (question) => {
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true); setStopped(false); setError(''); setInput('')
    try {
      const result = await api.ask({ question, scope, top_k: topK, use_reranker: true, history: turns.map((turn) => ({ question: turn.question, answer: turn.answer, context: turn.context, sources: turn.sources })) }, controller.signal)
      onUpdateSession(session.id, (current) => ({ ...current, qaTurns: [...(current.qaTurns || []), result] }))
      onHistory({ type: '规范问答', question, summary: result.answer?.conclusion || '' })
    } catch (requestError) {
      if (requestError.name === 'AbortError') setStopped(true)
      else setError(requestError.message)
    } finally { abortRef.current = null; setLoading(false) }
  }
  const stopGeneration = () => abortRef.current?.abort()
  return <div className="qa-workspace">
    <div className="qa-toolbar"><ScopeSelect library={library} value={scope} onChange={setScope} /><div className="toolbar-actions"><CitationCountControl value={topK} onChange={setTopK} />{turns.length > 0 && <button className="text-button" onClick={() => onUpdateSession(session.id, { qaTurns: [] })}>清空对话</button>}</div></div>
    <div className="conversation" ref={scrollRef}>{turns.length === 0 ? <div className="qa-empty"><div className="empty-icon"><ShieldCheck size={28} weight="fill" /></div><h2>直接询问建筑消防规范</h2><p>回答会附带标准、条文号、PDF 物理页和原文片段。证据不足时不会补写结论。</p><div className="example-list">{examples.map((example) => <button key={example} onClick={() => send(example)}>{example}<ArrowRight size={15} /></button>)}</div></div> : turns.map((turn, index) => <AnswerCard turn={turn} expert={expert} key={`${turn.question}-${index}`} />)}<GenerationControl loading={loading} stopped={stopped} onStop={stopGeneration} />{error && <div className="error-banner"><WarningCircle size={18} />{error}<button onClick={() => setError('')}><X size={15} /></button></div>}</div>
    <div className="composer-wrap"><Composer value={input} setValue={setInput} onSend={send} loading={loading} placeholder="输入建筑消防规范问题..." /></div>
  </div>
}

function Metric({ label, value, unit }) { return <div className="metric"><span>{label}</span><strong>{value ?? '-'}{unit && <em>{unit}</em>}</strong></div> }

function IFCAnswer({ item, kind, expert }) {
  if (kind === 'model') return <article className="ifc-answer"><div className="answer-heading"><Buildings size={18} weight="fill" /><span>模型助手</span></div><p className="conclusion">{cleanDisplayText(item.answer)}</p>{item.facts_used?.length > 0 && <p className="facts-used">使用字段：{item.facts_used.map(cleanDisplayText).join('；')}</p>}{item.missing_fields?.length > 0 && <div className="missing-fields"><WarningCircle size={16} />仍缺少：{item.missing_fields.join('、')}</div>}<span className={`confidence ${item.certainty || 'low'}`}>模型事实置信度：{confidenceLabel(item.certainty)}</span></article>
  const verdict = item.verdict || item.answer?.verdict
  return <div className="ifc-review-wrap">{verdict && <div className={`verdict-badge ${String(verdict).toLowerCase()}`}>合规结论：{verdict === 'PASS' ? '通过' : verdict === 'FAIL' ? '不通过' : '信息不足'}</div>}<AnswerCard turn={item} expert={expert} /></div>
}

function buildModelGraph(summary) {
  const storeys = summary?.storeys || []
  const spaces = summary?.spaces || []
  const doors = summary?.doors || []
  const windows = summary?.windows || []
  const stairs = (summary?.fire_related_elements || []).filter((item) => item.ifc_type === 'IfcStair')
  const rows = storeys.map((storey) => {
    const name = storey.name
    return {
      ...storey,
      spaces: spaces.filter((item) => item.storey === name),
      doors: doors.filter((item) => item.storey === name),
      windows: windows.filter((item) => item.storey === name),
      stairs: stairs.filter((item) => item.storey === name),
    }
  })
  const known = new Set(storeys.map((item) => item.name))
  return {
    rows,
    unmatched: {
      spaces: spaces.filter((item) => !known.has(item.storey)),
      doors: doors.filter((item) => !known.has(item.storey)),
      windows: windows.filter((item) => !known.has(item.storey)),
    },
  }
}

function ModelGraph({ rows }) {
  const groups = (rows || []).map((storey) => {
    const children = [
      ...(storey.spaces || []).slice(0, 5).map((item) => ({ kind: '空间', label: item.name, value: item.area })),
      ...(storey.doors || []).slice(0, 3).map((item) => ({ kind: '门', label: item.name, value: item.width_m })),
      ...(storey.windows || []).slice(0, 2).map((item) => ({ kind: '窗', label: item.name, value: item.width_m })),
      ...(storey.stairs || []).slice(0, 2).map((item) => ({ kind: '楼梯', label: item.name, value: item.storey })),
    ]
    return { storey, children }
  })
  const maxChildren = Math.max(1, ...groups.map((group) => group.children.length))
  const width = Math.max(760, groups.length * 190)
  const height = Math.max(180, 100 + maxChildren * 34)
  return <svg viewBox={`0 0 ${width} ${height}`} className="model-graph-svg" role="img" aria-label="IFC 模型空间关系图谱">
    {groups.map((group, groupIndex) => {
      const x = 60 + groupIndex * 190
      const floorY = 40
      const childStartY = 92
      return <g key={`${group.storey.name}-${groupIndex}`}>
        {group.children.map((child, childIndex) => {
          const cy = childStartY + childIndex * 34
          return <g key={`${child.kind}-${child.label}-${childIndex}`}><line x1={x} y1={floorY + 12} x2={x} y2={cy - 8} className="graph-edge" /><circle cx={x} cy={cy} r={6} className="graph-node" /><text x={x + 13} y={cy + 3} className="graph-label">{child.label}</text></g>
        })}
        <rect x={x - 42} y={floorY - 13} width={84} height={26} rx={4} className="graph-floor" />
        <text x={x} y={floorY + 4} textAnchor="middle" className="graph-floor-label">{group.storey.name}</text>
      </g>
    })}
  </svg>
}

function BIMWorkspace({ library, expert, onHistory, session, onUpdateSession, sessions, activeSessionId, onCreateSession, onSelectSession }) {
  const [scope, setScope] = useState('auto'), [parsing, setParsing] = useState(false)
  const [topK, setTopK] = useState(5)
  const [tab, setTab] = useState('overview'), [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false), [preparing, setPreparing] = useState(false)
  const [batchProgress, setBatchProgress] = useState(null)
  const [stopped, setStopped] = useState(false), [error, setError] = useState('')
  const [suggestionIndex, setSuggestionIndex] = useState(0)
  const abortRef = useRef(null)
  const reviewScrollRef = useRef(null)
  const data = session.ifcData || null
  const file = session.ifcFileName || ''
  const fileSize = session.ifcFileSize || 0
  const ifcModels = session.ifcModels || []
  const libraryModels = ifcModels.length ? ifcModels : (file ? [{ id: session.ifcActiveModelId || 'current', fileName: file, fileSize, summary }] : [])
  const projectContext = session.projectContext || ''
  const modelAnswers = session.modelAnswers || []
  const reviewAnswers = session.reviewAnswers || []
  const projectProfile = session.projectProfile || null
  const reviewPreparationStatus = session.reviewPreparationStatus || ''
  const profileIsCurrent = Boolean(projectProfile) && session.projectProfileContext === projectContext
  const reviewQuestions = session.reviewQuestions?.length ? session.reviewQuestions : (data?.review_questions || [])
  const summary = data?.summary, counts = summary?.counts || {}
  const modelGraph = useMemo(() => buildModelGraph(summary), [summary])
  useEffect(() => { setTab('overview'); setQuestion('') }, [session.id])
  useEffect(() => { document.querySelector('.bim-workspace')?.scrollTo({ top: 0 }) }, [tab])
  useEffect(() => { reviewScrollRef.current?.scrollTo({ top: reviewScrollRef.current.scrollHeight, behavior: 'smooth' }) }, [reviewAnswers, loading])
  const uploadFiles = async (files) => {
    const targets = Array.from(files || [])
    if (!targets.length) return
    setParsing(true); setError(''); setBatchProgress({ current: 0, total: targets.length, name: '' })
    try {
      for (let index = 0; index < targets.length; index += 1) {
        const selected = targets[index]
        setBatchProgress({ current: index + 1, total: targets.length, name: selected.name })
        const parsed = await api.parseIfc(selected)
        let preparation = null
        try {
          preparation = await api.prepareIfcReview({ summary: parsed.summary, project_context: '' })
        } catch { /* Parsing remains usable when suggestion preparation is temporarily unavailable. */ }
        const entry = {
          id: `ifc-${Date.now()}-${index}-${Math.random().toString(36).slice(2, 6)}`,
          fileName: selected.name,
          fileSize: selected.size,
          summary: parsed.summary,
          projectContext: '',
          projectProfile: preparation?.project_profile || null,
          projectProfileContext: '',
          reviewQuestions: preparation?.review_questions || parsed.review_questions || [],
          reviewPreparationStatus: preparation?.preparation_status || 'fallback',
          modelAnswers: [],
          reviewAnswers: [],
          projectDate: '',
        }
        onUpdateSession(session.id, (current) => {
          const models = [...(current.ifcModels || []), entry]
          return {
            ...current,
            ifcModels: models,
            ifcActiveModelId: entry.id,
            ifcData: { summary: parsed.summary, review_questions: parsed.review_questions },
            ifcFileName: entry.fileName,
            ifcFileSize: entry.fileSize,
            projectContext: '',
            projectProfile: entry.projectProfile,
            projectProfileContext: '',
            reviewQuestions: entry.reviewQuestions,
            reviewPreparationStatus: entry.reviewPreparationStatus,
            modelAnswers: [],
            reviewAnswers: [],
            projectDate: '',
          }
        })
      }
      setTab('overview')
    } catch (e) { setError(e.message) } finally { setParsing(false); setBatchProgress(null) }
  }
  const selectModel = (id) => {
    onUpdateSession(session.id, (current) => {
      const models = (current.ifcModels || []).map((model) => model.id === current.ifcActiveModelId
        ? { ...model, projectContext: current.projectContext, projectProfile: current.projectProfile, projectProfileContext: current.projectProfileContext, reviewQuestions: current.reviewQuestions, reviewPreparationStatus: current.reviewPreparationStatus, modelAnswers: current.modelAnswers, reviewAnswers: current.reviewAnswers, projectDate: current.projectDate }
        : model)
      const target = models.find((model) => model.id === id)
      if (!target) return current
      return {
        ...current,
        ifcModels: models,
        ifcActiveModelId: target.id,
        ifcData: { summary: target.summary, review_questions: target.reviewQuestions || target.summary?.review_questions || [] },
        ifcFileName: target.fileName,
        ifcFileSize: target.fileSize,
        projectContext: target.projectContext || '',
        projectProfile: target.projectProfile || null,
        projectProfileContext: target.projectProfileContext || '',
        reviewQuestions: target.reviewQuestions || [],
        reviewPreparationStatus: target.reviewPreparationStatus || '',
        modelAnswers: target.modelAnswers || [],
        reviewAnswers: target.reviewAnswers || [],
        projectDate: target.projectDate || '',
      }
    })
    setTab('overview'); setQuestion('')
  }
  const deleteModel = (id) => {
    if (!window.confirm('删除该已解析模型？')) return
    onUpdateSession(session.id, (current) => {
      const hasModels = Array.isArray(current.ifcModels) && current.ifcModels.length > 0
      if (!hasModels) {
        return { ...current, ifcData: null, ifcFileName: '', ifcFileSize: 0, ifcActiveModelId: '', projectContext: '', projectProfile: null, projectProfileContext: '', reviewQuestions: [], reviewPreparationStatus: '', modelAnswers: [], reviewAnswers: [], projectDate: '' }
      }
      const models = current.ifcModels.map((model) => model.id === current.ifcActiveModelId
        ? { ...model, projectContext: current.projectContext, projectProfile: current.projectProfile, projectProfileContext: current.projectProfileContext, reviewQuestions: current.reviewQuestions, reviewPreparationStatus: current.reviewPreparationStatus, modelAnswers: current.modelAnswers, reviewAnswers: current.reviewAnswers, projectDate: current.projectDate }
        : model).filter((model) => model.id !== id)
      const target = models[0]
      if (!target) {
        return { ...current, ifcModels: [], ifcActiveModelId: '', ifcData: null, ifcFileName: '', ifcFileSize: 0, projectContext: '', projectProfile: null, projectProfileContext: '', reviewQuestions: [], reviewPreparationStatus: '', modelAnswers: [], reviewAnswers: [], projectDate: '' }
      }
      return {
        ...current,
        ifcModels: models,
        ifcActiveModelId: target.id,
        ifcData: { summary: target.summary, review_questions: target.reviewQuestions || target.summary?.review_questions || [] },
        ifcFileName: target.fileName,
        ifcFileSize: target.fileSize,
        projectContext: target.projectContext || '',
        projectProfile: target.projectProfile || null,
        projectProfileContext: target.projectProfileContext || '',
        reviewQuestions: target.reviewQuestions || [],
        reviewPreparationStatus: target.reviewPreparationStatus || '',
        modelAnswers: target.modelAnswers || [],
        reviewAnswers: target.reviewAnswers || [],
        projectDate: target.projectDate || '',
      }
    })
    setTab('overview'); setQuestion('')
  }
  const prepareReview = async () => {
    if (!summary) return null
    setPreparing(true); setError('')
    try {
      const result = await api.prepareIfcReview({ summary, project_context: projectContext })
      onUpdateSession(session.id, {
        projectProfile: result.project_profile,
        projectProfileContext: projectContext,
        reviewQuestions: result.review_questions || [],
        reviewPreparationStatus: result.preparation_status || 'generated',
      })
      return result
    } catch (e) {
      setError(e.message)
      return null
    } finally { setPreparing(false) }
  }
  const suggestReviewQuestion = async () => {
    let questions = reviewQuestions
    if (!questions.length && summary) {
      const prepared = await prepareReview()
      questions = prepared?.review_questions || reviewQuestions
    }
    if (!questions.length) return
    const item = questions[suggestionIndex % questions.length]
    if (item?.question) {
      setQuestion(item.question)
      setSuggestionIndex((index) => index + 1)
    }
  }
  const clearReviewAnswers = () => onUpdateSession(session.id, { reviewAnswers: [] })
  const verdictLabel = (item) => (item.verdict === 'PASS' ? '通过' : item.verdict === 'FAIL' ? '不通过' : '信息不足')
  const passCount = reviewAnswers.filter((item) => item.verdict === 'PASS').length
  const failCount = reviewAnswers.filter((item) => item.verdict === 'FAIL').length
  const insufficientCount = reviewAnswers.length - passCount - failCount
  const exportReport = () => {
    const reportId = `R-${Date.now()}`
    const lines = ['# 消防合规审查报告', '', `- 报告编号：${reportId}`, '- 报告状态：草稿/待人工复核', `- 模型文件：${file || '-'}`, `- 生成时间：${new Date().toLocaleString()}`, `- IFC Schema：${summary?.project?.schema || '-'}`, `- 项目单位：${summary?.project_units?.length_label || 'm'}${summary?.project_units?.length_label && summary.project_units.length_label !== 'm' ? '（已换算为 m）' : ''}`, `- 楼层：${counts.IfcBuildingStorey || 0}`, `- 空间：${counts.IfcSpace || 0}`, `- 门：${counts.IfcDoor || 0}`, `- 楼梯：${counts.IfcStair || 0}`, `- 项目日期：${session.projectDate || '未填写'}`]
    if (summary?.building_use?.value) lines.push(`- 建筑用途推断：${summary.building_use.value}（置信度：${summary.building_use.confidence || '低'}）`)
    if (projectContext) lines.push(`- 项目条件补充：${projectContext}`)
    if (summary?.field_quality?.available?.length) lines.push(`- 可用字段：${summary.field_quality.available.join('、')}`)
    if (summary?.field_quality?.missing?.length) lines.push(`- 缺失字段：${summary.field_quality.missing.join('、')}`)
    if (summary?.unit_quality?.issues?.length) lines.push(`- 单位异常：${summary.unit_quality.issues.join('；')}`)
    lines.push('', '## 结论汇总', `- 通过：${passCount}`, `- 不通过：${failCount}`, `- 信息不足：${insufficientCount}`)
    lines.push('', '## 模型事实问答')
    if (!modelAnswers.length) lines.push('（无）')
    modelAnswers.forEach((item, index) => {
      lines.push(`### ${index + 1}. ${item.question}`, '', `回答：${item.answer || '未生成'}`, '')
      if (item.missing_fields?.length) lines.push(`仍需确认：${item.missing_fields.join('、')}`, '')
    })
    lines.push('', '## 消防合规审查')
    if (!reviewAnswers.length) lines.push('（无）')
    reviewAnswers.forEach((item, index) => {
      const answer = item.answer || {}
      lines.push(`### ${index + 1}. ${item.question}`, '', `合规结论：${verdictLabel(item)}`, `结论：${answer.conclusion || '未生成'}`)
      if (answer.missing_fields?.length) lines.push(`仍需确认：${answer.missing_fields.join('、')}`)
      if (item.sources?.length) {
        lines.push('引用来源：')
        item.sources.forEach((source) => lines.push(`- ${source.source_file || source.standard_id || '未识别'}${source.article_no ? ` ${source.article_no}` : ''}${source.page != null ? `，PDF 物理页第 ${source.page} 页` : ''}`))
      }
      lines.push('')
    })
    lines.push('', '## 免责声明', '本报告由系统辅助生成，不能替代正式消防审查、审图机构或主管部门的最终认定；所有结论均需人工复核。')
    downloadTextFile(`${(file || 'ifc').replace(/[\\/:*?"<>|]/g, '_').slice(0, 80)}_审查报告_${reportId}.md`, lines.join('\n'))
  }
  const exportReportJson = () => {
    const reportId = `R-${Date.now()}`
    const payload = {
      report_id: reportId,
      report_type: 'ifc_fire_compliance_review',
      report_status: 'draft_pending_human_review',
      created_at: new Date().toISOString(),
      model: {
        file,
        schema: summary?.project?.schema,
        project_units: summary?.project_units,
        counts,
        building_use: summary?.building_use,
        field_quality: summary?.field_quality,
        unit_quality: summary?.unit_quality,
        space_status: summary?.space_status,
      },
      project_context: projectContext,
      project_date: session.projectDate || '',
      model_answers: modelAnswers.map((item) => ({ question: item.question, answer: item.answer, missing_fields: item.missing_fields || [] })),
      review_answers: reviewAnswers.map((item) => ({
        question: item.question,
        verdict: item.verdict,
        conclusion: item.answer?.conclusion || '',
        missing_fields: item.answer?.missing_fields || [],
        sources: item.sources || [],
      })),
      conclusion_summary: { pass: passCount, fail: failCount, insufficient: insufficientCount },
    }
    downloadTextFile(`${(file || 'ifc').replace(/[\\/:*?"<>|]/g, '_').slice(0, 80)}_审查档案_${reportId}.json`, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8')
  }
  const sendModel = async (text) => {
    const controller = new AbortController(); abortRef.current = controller
    setLoading(true); setStopped(false); setError(''); setQuestion('')
    try {
      const result = await api.askIfcModel({ summary, question: text, project_context: projectContext }, controller.signal)
      onUpdateSession(session.id, (current) => ({ ...current, modelAnswers: [...(current.modelAnswers || []), { question: text, ...result }] }))
      onHistory({ type: '模型查询', question: text, summary: result.answer || '' })
    } catch (e) {
      if (e.name === 'AbortError') setStopped(true); else setError(e.message)
    } finally { abortRef.current = null; setLoading(false) }
  }
  const sendReview = async (text) => {
    const controller = new AbortController(); abortRef.current = controller
    setLoading(true); setStopped(false); setError(''); setQuestion('')
    try {
      let activeProfile = projectProfile
      if (!profileIsCurrent) {
        try {
          const prepared = await api.prepareIfcReview({ summary, project_context: projectContext }, controller.signal)
          activeProfile = prepared.project_profile
          onUpdateSession(session.id, {
            projectProfile: prepared.project_profile,
            projectProfileContext: projectContext,
            reviewQuestions: prepared.review_questions || [],
            reviewPreparationStatus: prepared.preparation_status || 'generated',
          })
        } catch (e) {
          if (e.name === 'AbortError') throw e
          activeProfile = null
        }
      }
      const result = await api.askIfcCompliance({ summary, question: text, project_context: projectContext, project_profile: activeProfile || {}, scope, top_k: topK, use_reranker: true, applicable_date: session.projectDate || '' }, controller.signal)
      onUpdateSession(session.id, (current) => ({ ...current, reviewAnswers: [...(current.reviewAnswers || []), result] }))
      onHistory({ type: '消防合规审查', question: text, summary: result.answer?.conclusion || '' })
    } catch (e) {
      if (e.name === 'AbortError') setStopped(true); else setError(e.message)
    } finally { abortRef.current = null; setLoading(false) }
  }
  const stopGeneration = () => abortRef.current?.abort()
  return <div className="bim-workspace">
    {tab !== 'review' && <div className="bim-top-grid">
      <section className="upload-panel"><div className="section-title"><div><h2>模型来源</h2><p>上传 IFC 后读取模型事实</p></div></div><label className={`dropzone ${file ? 'has-file' : ''}`}><input type="file" accept=".ifc,.ifcxml" multiple onChange={(event) => uploadFiles(event.target.files)} />{parsing ? <SpinnerGap size={28} className="spin" /> : file ? <CheckCircle size={28} weight="fill" /> : <UploadSimple size={28} />}<strong>{parsing ? '正在解析模型' : file ? file : '上传 IFC 模型'}</strong><span>{file ? `${(fileSize / 1024 / 1024).toFixed(1)} MB` : '支持 .ifc / .ifcxml，最大 200MB，可多选'}</span></label>{batchProgress && <div className="batch-progress"><SpinnerGap size={14} className="spin" />正在解析 {batchProgress.current}/{batchProgress.total}：{batchProgress.name}</div>}</section>
      <section className="summary-panel"><div className="section-title"><div><h2>模型事实摘要</h2><p>{summary ? '解析完成，以下数值来自 IFC 模型' : '等待上传模型'}</p></div>{summary && <span className="synced"><CheckCircle size={15} weight="fill" />已同步</span>}</div><div className="metrics-grid"><Metric label="楼层" value={counts.IfcBuildingStorey} /><Metric label="空间" value={counts.IfcSpace} /><Metric label="门" value={counts.IfcDoor} /><Metric label="楼梯" value={counts.IfcStair} /><Metric label="推算高度" value={summary?.building_height_m} unit="m" /></div></section>
    </div>}
    {tab !== 'review' && libraryModels.length > 0 && <section className="model-library-panel"><div className="section-title"><div><h2>已解析模型</h2><p>批量解析后可切换当前审查模型，每个模型独立保存项目条件与问答记录</p></div><span>{libraryModels.length} 个模型</span></div><div className="model-library-grid">{libraryModels.map((model) => { const active = model.id === session.ifcActiveModelId || (!ifcModels.length && model.id === 'current'); return <article className={`model-card ${active ? 'active' : ''}`} key={model.id}><div className="model-card-head"><strong title={model.fileName}>{model.fileName}</strong><div className="model-card-actions">{active ? <span className="model-active-badge">当前</span> : <button className="text-button" onClick={() => selectModel(model.id)} type="button">切换</button>}<button className="icon-button" onClick={() => deleteModel(model.id)} title="删除模型" type="button"><Trash size={14} /></button></div></div><div className="model-card-meta"><span>{model.summary?.counts?.IfcBuildingStorey || 0} 层</span><span>{model.summary?.counts?.IfcDoor || 0} 门</span><span>{model.summary?.counts?.IfcStair || 0} 楼梯</span><span>高度 {model.summary?.building_height_m ?? '-'} m</span></div><small className="model-card-file">{model.fileSize ? `${(model.fileSize / 1024 / 1024).toFixed(1)} MB` : ''}</small></article> })}</div></section>}
    {error && <div className="error-banner max-panel"><WarningCircle size={18} />{error}<button onClick={() => setError('')}><X size={15} /></button></div>}
    {summary ? <>
      {tab !== 'review' && <section className="project-context">
        <div className="section-title"><div><h2>项目条件补充</h2><p>补充模型中没有、但影响规范适用范围的真实项目条件</p></div><span>自然语言输入</span></div>
        <textarea value={projectContext} onChange={(event) => onUpdateSession(session.id, { projectContext: event.target.value })} placeholder="例如：多层住宅，共 4 层，无地下室，设置自动喷水灭火系统；门宽为模型净宽..." />
        <label className="project-date-field"><span>项目日期</span><input type="date" value={session.projectDate || ''} onChange={(event) => onUpdateSession(session.id, { projectDate: event.target.value })} /></label>
        <div className="project-context-actions">
          <span className={profileIsCurrent ? 'context-status current' : 'context-status'}>{profileIsCurrent ? '当前条件已用于审查' : '条件有更新，发送问题时会自动应用'}</span>
          <button className="secondary-button" onClick={prepareReview} disabled={preparing}>{preparing ? <SpinnerGap size={15} className="spin" /> : <CheckCircle size={15} />}应用项目条件</button>
        </div>
        {projectProfile && profileIsCurrent && <div className="project-profile-summary">
          {projectProfile.confirmed_conditions?.length > 0 && <div><strong>已确认</strong><span>{projectProfile.confirmed_conditions.join('；')}</span></div>}
          {projectProfile.ifc_facts?.length > 0 && <div><strong>模型事实</strong><span>{projectProfile.ifc_facts.slice(0, 4).join('；')}</span></div>}
          {projectProfile.unresolved_conditions?.length > 0 && <div><strong>待补充</strong><span>{projectProfile.unresolved_conditions.join('；')}</span></div>}
        </div>}
      </section>}
      <div className="subtabs" role="tablist">{[['overview', '模型概览'], ['model', '模型内容查询'], ['review', '消防合规审查'], ['fields', '解析字段']].map(([id, label]) => <button key={id} className={tab === id ? 'active' : ''} onClick={() => { setTab(id); setQuestion('') }}>{label}</button>)}</div>
      {tab === 'overview' && <section className="tab-panel overview-grid"><div><h3>项目信息</h3><dl><dt>文件</dt><dd>{summary.source_file}</dd><dt>IFC Schema</dt><dd>{summary.project?.schema || '-'}</dd><dt>项目</dt><dd>{summary.project?.project_name || '-'}</dd><dt>建筑</dt><dd>{summary.project?.building_name || '未命名'}</dd></dl></div><div><h3>可审查性</h3><p className="reviewability">{summary.field_quality?.reviewability}</p><div className="tag-list">{(summary.field_quality?.available || []).map((item) => <span className="tag good" key={item}>{item}</span>)}</div><div className="tag-list">{(summary.field_quality?.missing || []).map((item) => <span className="tag missing" key={item}>{item}</span>)}</div></div><div className="wide"><h3>建筑用途识别</h3><p>{summary.building_use?.value} <span className="confidence low">置信度：{summary.building_use?.confidence || '低'}</span></p><small>{summary.building_use?.reason}</small></div>{summary.space_status?.status === 'not_defined' && <div className="wide space-status-note"><h3>空间状态</h3><p>{summary.space_status.note}</p></div>}<div className="wide"><h3>单位与数值质量</h3><p className="reviewability">长度：m；面积：㎡；高程：m{summary.project_units?.length_label && summary.project_units.length_label !== 'm' ? `（IFC 原始单位：${summary.project_units.length_label}，已换算）` : ''}</p>{(summary.unit_quality?.issues || []).length ? <div className="tag-list">{(summary.unit_quality?.issues || []).map((item) => <span className="tag missing" key={item}>{item}</span>)}</div> : <p className="unit-ok">未发现明显异常</p>}</div></section>}
      {tab === 'model' && <section className="tab-panel query-panel"><div className="query-head"><div><h3>模型内容查询</h3><p>只回答 IFC 中能读取或计算的事实，不进入规范检索。</p></div><Buildings size={24} /></div><div className="answer-stream">{modelAnswers.map((item, index) => <div key={`${item.question}-${index}`}><div className="compact-question">{item.question}</div><IFCAnswer item={item} kind="model" expert={expert} /></div>)}<GenerationControl loading={loading} stopped={stopped} onStop={stopGeneration} /></div><Composer value={question} setValue={setQuestion} onSend={sendModel} loading={loading} placeholder="询问模型事实，例如：这栋建筑的平均楼层高度是多少？" /></section>}
      {tab === 'review' && <section className="tab-panel review-panel">
        <div className="review-head">
          <div className="query-head"><div><h3>消防合规审查</h3><p>将 IFC 事实、项目条件和规范原文共同用于判断。</p></div><ShieldCheck size={24} /></div>
          <div className="review-head-actions"><button className="secondary-button report-export" onClick={exportReport} disabled={!reviewAnswers.length && !modelAnswers.length} type="button"><DownloadSimple size={15} />导出报告</button><button className="secondary-button report-export" onClick={exportReportJson} disabled={!reviewAnswers.length && !modelAnswers.length} type="button"><FileArrowUp size={15} />导出档案</button><SessionSwitch sessions={sessions} activeSessionId={activeSessionId} onSelect={onSelectSession} onCreate={onCreateSession} onClear={clearReviewAnswers} /></div>
        </div>
        <div className="review-controls"><ScopeSelect library={library} value={scope} onChange={setScope} /><div className="toolbar-actions"><CitationCountControl value={topK} onChange={setTopK} /></div></div>
        <div className="review-suggestions">
          <div className="suggestion-heading">
            <div><h4>建议复核项</h4><p>{reviewPreparationStatus === 'fallback' ? '智能建议暂不可用，当前显示按模型字段生成的备用复核项。' : '基于当前模型事实与项目条件动态生成，仅作为审查入口。'}</p></div>
            <div className="suggestion-actions">
              <button className="suggestion-action primary" onClick={suggestReviewQuestion} disabled={preparing} type="button"><DiceFive size={15} />给我一个问题</button>
              <button className="suggestion-action" onClick={() => prepareReview()} disabled={preparing || !summary} type="button">{preparing ? <SpinnerGap size={15} className="spin" /> : <ArrowsClockwise size={15} />}换一批</button>
            </div>
          </div>
          <div className="review-context">
            <span title={file}><FileText size={13} />{file}</span>
            <span>{counts.IfcBuildingStorey} 层</span>
            <span>{counts.IfcDoor} 门</span>
            <span>{counts.IfcStair} 楼梯</span>
            {session.projectDate && <span title={session.projectDate}>项目日期：{session.projectDate}</span>}
            {projectContext && <span className="review-context-project" title={projectContext}>项目条件：{projectContext}</span>}
          </div>
          <div className="suggestion-strip">
            {reviewQuestions.length ? reviewQuestions.map((item, index) => <button className="suggestion-chip" key={`${item.title}-${index}`} onClick={() => setQuestion(item.question)} title={item.reason || item.question} type="button"><span>{item.title}</span><ArrowRight size={14} /></button>) : <span className="suggestion-empty">{preparing ? '正在生成建议...' : '暂无建议，点击“给我一个问题”生成。'}</span>}
          </div>
        </div>
        <div className="review-conversation" ref={reviewScrollRef}>
          {reviewAnswers.length === 0 ? <div className="review-empty"><ChatCircleDots size={30} /><p>从当前模型事实和项目条件出发，生成可继续追问的消防审查问题。</p><button onClick={suggestReviewQuestion} disabled={preparing} type="button"><DiceFive size={15} />给我一些问题</button></div> : reviewAnswers.map((item, index) => <IFCAnswer item={item} kind="review" expert={expert} key={`${item.question}-${index}`} />)}
          <GenerationControl loading={loading} stopped={stopped} onStop={stopGeneration} />
        </div>
        <div className="composer-wrap"><Composer value={question} setValue={setQuestion} onSend={sendReview} loading={loading} placeholder="输入针对当前模型的消防审查问题..." /></div>
      </section>}
      {tab === 'fields' && <section className="tab-panel fields-panel"><h3>解析字段</h3><p>用于确认模型中实际存在的数据，不代表合规结论。</p><div className="field-columns"><div><h4>楼层（{summary.storeys?.length || 0}）</h4>{(summary.storeys || []).slice(0, 8).map((item, i) => <div className="field-row" key={i}><span>{item.name}</span><strong>{item.elevation_m ?? '-'} m</strong></div>)}</div><div><h4>门（{summary.doors?.length || 0}）</h4>{(summary.doors || []).slice(0, 8).map((item, i) => <div className="field-row" key={i}><span>{item.name}</span><strong>{item.width_m ?? '-'} m</strong></div>)}</div><div><h4>空间（{summary.spaces?.length || 0}）</h4>{(summary.spaces || []).slice(0, 8).map((item, i) => <div className="field-row" key={i}><span>{item.name}</span><strong>{item.area ?? '-'}</strong></div>)}</div></div>
      <div className="model-relations"><h4>模型空间关系</h4><p>由 IFC 楼层、空间、门、窗和楼梯等对象关系生成，不预设建筑类型。</p><div className="relation-tree">{modelGraph.rows.map((storey) => <details className="relation-node" key={storey.name} open><summary><strong>{storey.name}</strong><span>{storey.spaces.length} 空间 · {storey.doors.length} 门 · {storey.windows.length} 窗 · {storey.stairs.length} 楼梯</span></summary><div className="relation-children">{storey.spaces.length > 0 && <div><h5>空间</h5>{(storey.spaces || []).slice(0, 12).map((item, i) => <div className="field-row" key={`${item.global_id}-${i}`}><span>{item.name}</span><strong>{item.area ?? '-'}</strong></div>)}</div>}{storey.doors.length > 0 && <div><h5>门</h5>{(storey.doors || []).slice(0, 8).map((item, i) => <div className="field-row" key={`${item.global_id}-${i}`}><span>{item.name}</span><strong>{item.width_m ?? '-'} m</strong></div>)}</div>}{storey.windows.length > 0 && <div><h5>窗</h5>{(storey.windows || []).slice(0, 6).map((item, i) => <div className="field-row" key={`${item.global_id}-${i}`}><span>{item.name}</span><strong>{item.width_m ?? '-'} m</strong></div>)}</div>}{storey.stairs.length > 0 && <div><h5>楼梯</h5>{(storey.stairs || []).slice(0, 6).map((item, i) => <div className="field-row" key={`${item.name}-${i}`}><span>{item.name}</span><strong>{item.storey || '-'}</strong></div>)}</div>}</div></details>)}{(modelGraph.unmatched.spaces.length > 0 || modelGraph.unmatched.doors.length > 0 || modelGraph.unmatched.windows.length > 0) && <details className="relation-node unmatched" key="unmatched"><summary><strong>未归属楼层</strong><span>{modelGraph.unmatched.spaces.length} 空间 · {modelGraph.unmatched.doors.length} 门 · {modelGraph.unmatched.windows.length} 窗</span></summary><div className="relation-children">{(modelGraph.unmatched.spaces || []).slice(0, 12).map((item, i) => <div className="field-row" key={`space-${item.global_id}-${i}`}><span>{item.name}</span><strong>{item.area ?? '-'}</strong></div>)}{(modelGraph.unmatched.doors || []).slice(0, 8).map((item, i) => <div className="field-row" key={`door-${item.global_id}-${i}`}><span>{item.name}</span><strong>{item.width_m ?? '-'} m</strong></div>)}</div></details>}</div></div><div className="model-graph-card"><h4>模型关系图谱</h4><p>由 IFC 楼层与对象包含关系生成，用于快速查看空间归属，不预设建筑类型。</p><ModelGraph rows={modelGraph.rows} /></div></section>}
    </> : <div className="bim-empty"><FileArrowUp size={30} /><h2>先上传一个 IFC 模型</h2><p>解析完成后，可分别查询模型事实或结合规范进行消防合规审查。</p></div>}
  </div>
}

function LibraryWorkspace({ library }) {
  const [search, setSearch] = useState('')
  const rows = useMemo(() => (library?.standards || []).filter((item) => `${item.title}${item.standard_id}${item.source_file}`.toLowerCase().includes(search.toLowerCase())), [library, search])
  return <div className="support-page"><div className="page-actions"><div><h2>已收录规范</h2><p>知识库范围来自当前已入库文件，不使用固定分类名称。</p></div><label className="search-box"><MagnifyingGlass size={17} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="搜索文件或标准号" /></label></div><div className="library-table"><div className="table-head"><span>规范文件</span><span>标准号</span><span>知识片段</span><span>状态</span></div>{rows.map((item) => <div className="table-row" key={`${item.standard_id}-${item.source_file}`}><span><FileText size={18} /><div><strong>{item.title}</strong><small>{item.source_file}</small></div></span><span>{item.standard_id}</span><span>{item.chunks}</span><span className={item.status === '现行有效' ? 'valid' : 'pending'}>{item.status}</span></div>)}</div></div>
}

function VersionTimeline({ topic }) {
  const versions = topic.versions || []
  const latest = versions[versions.length - 1]
  const kindLabel = topic.topic_kind === 'generated' ? '动态生成主题' : topic.topic_kind === 'retrieval' ? '检索版本候选' : '原文标注示例'
  return <div className="version-detail-content">
    <div className="version-detail-head">
      <div><span className="version-eyebrow">{kindLabel}</span><h3>{cleanDisplayText(topic.title)}</h3><p>{cleanDisplayText(topic.subject)}</p></div>
      <div className="version-range">{versions[0]?.effective_date}<ArrowsLeftRight size={14} />{latest?.effective_date}</div>
    </div>
    {topic.change_summary && <div className="change-summary"><ArrowsLeftRight size={17} /><div><strong>变化要点</strong><p><VersionHighlight text={topic.change_summary} /></p></div></div>}
    {topic.applicability_note && <div className="version-note"><strong>适用提示</strong><p><VersionHighlight text={topic.applicability_note} /></p></div>}
    {topic.replaced_article_count != null && <div className="version-dynamic-note"><strong>自动生成范围</strong><span>覆盖 {topic.replaced_article_count} 条被替代条文{topic.sample_articles?.length ? `，示例：${topic.sample_articles.slice(0, 8).join('、')}` : ''}</span></div>}
    <div className="version-timeline">
      {versions.map((item, index) => <div className="version-item" key={`${item.standard_id}-${item.article_no}-${index}`}>
        <div className="version-axis"><span>{index + 1}</span>{index < versions.length - 1 && <i />}</div>
        <article className={`version-card ${index === versions.length - 1 ? 'latest' : ''}`}>
          <div className="version-card-head"><div><strong>{item.standard_id}</strong><span>{item.title}</span></div><div className="version-status-wrap"><span className={`version-status ${item.status}`}>{item.status_label}</span>{item.applicable === false && <span className="version-status not-applicable">晚于项目日期</span>}</div></div>
          <div className="version-meta"><span>施行 {item.effective_date}</span><span>{cleanDisplayText(item.article_no)}</span>{item.page && <span>{/^\d+/.test(item.page) ? `PDF 物理页第 ${item.page} 页` : cleanDisplayText(item.page)}</span>}</div>
          <p className="version-requirement"><VersionHighlight text={item.requirement} /></p>
          {item.source_excerpt && <details className="version-excerpt"><summary>查看原文片段</summary><p><VersionHighlight text={item.source_excerpt} /></p></details>}
        </article>
      </div>)}
    </div>
  </div>
}

function VersionWorkspace() {
  const [examples, setExamples] = useState([])
  const [query, setQuery] = useState('')
  const [projectDate, setProjectDate] = useState('')
  const [active, setActive] = useState(null)
  const [resolving, setResolving] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    api.versionTopics().then((data) => {
      setExamples((data.topics || []).filter((item) => item.topic_kind === 'curated').slice(0, 4))
    }).catch(() => { /* Examples are optional; query remains the primary entry. */ })
  }, [])
  const resolve = async (text) => {
    const value = (text ?? query).trim()
    if (!value) return
    setQuery(value)
    setResolving(true); setError('')
    try {
      const data = await api.versionResolve({ question: value, top_k: 8, applicable_date: projectDate })
      setActive(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setResolving(false)
    }
  }
  return <div className="version-workspace">
    <div className="page-actions"><div><h2>规范版本对比</h2><p>输入一个审查主题，系统把不同版本和不同规范对同一主题的条文拉出来；这里不是固定主题目录。</p></div></div>
    {error && <div className="error-banner max-panel"><WarningCircle size={18} />{error}<button onClick={() => setError('')}><X size={15} /></button></div>}
    <div className="version-query-panel">
      <form className="version-query-form" onSubmit={(event) => { event.preventDefault(); resolve() }}>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入要对比的主题，例如：住宅户门净宽" />
        <label className="version-date-label"><span>项目日期</span><input type="date" value={projectDate} onChange={(event) => setProjectDate(event.target.value)} /></label>
        <button type="submit" disabled={resolving || !query.trim()}>{resolving ? <SpinnerGap size={17} className="spin" /> : <MagnifyingGlass size={17} />}查询版本沿革</button>
      </form>
      {examples.length > 0 && <div className="version-examples"><span>输入示例</span>{examples.map((topic) => <button key={topic.topic_id} onClick={() => resolve(topic.title)} type="button">{topic.title}</button>)}</div>}
    </div>
    <section className="version-detail">
      {resolving ? <div className="version-empty"><SpinnerGap size={24} className="spin" /><p>正在检索不同版本的规范条文...</p></div> : active ? <VersionTimeline topic={active} /> : <div className="version-empty"><GitBranch size={30} /><p>输入一个主题后，这里会按时间展示相关规范版本</p></div>}
    </section>
  </div>
}

function HistoryWorkspace({ history, clearHistory }) { return <div className="support-page"><div className="page-actions"><div><h2>查询历史</h2><p>保留本次使用期间的问答与审查记录。</p></div>{history.length > 0 && <button className="secondary-button" onClick={clearHistory}>清空记录</button>}</div>{history.length ? <div className="history-list">{history.map((item, index) => <article key={`${item.question}-${index}`}><span>{item.type}</span><h3>{item.question}</h3><p>{item.summary}</p></article>)}</div> : <div className="support-empty"><ClockCounterClockwise size={30} /><p>还没有查询记录</p></div>}</div> }

function StatusWorkspace({ health, library }) {
  const [metrics, setMetrics] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('fire-review-token') || '')
  useEffect(() => { api.statusMetrics().then(setMetrics).catch(() => {}) }, [])
  const saveToken = () => { localStorage.setItem('fire-review-token', token.trim()); setToken(token.trim()) }
  const http = metrics?.metrics?.http
  const llm = metrics?.metrics?.llm
  const avgHttp = http?.requests ? (http.total_duration_ms / http.requests).toFixed(1) : '0'
  const avgLlm = llm?.calls ? (llm.total_duration_ms / llm.calls).toFixed(1) : '0'
  return <div className="support-page"><div className="page-actions"><div><h2>系统状态</h2><p>这里只展示当前服务可用性、运行观测和可选访问控制，不展示测试集和迭代过程。</p></div></div><div className="status-grid"><div><span className={`status-dot ${health?.status === 'ok' ? 'online' : ''}`} /><h3>后端服务</h3><strong>{health?.status === 'ok' ? '运行正常' : '未连接'}</strong></div><div><Database size={21} /><h3>规范知识库</h3><strong>{library?.standards?.length || 0} 份规范</strong></div><div><Sparkle size={21} /><h3>DeepSeek</h3><strong>{health?.deepseek ? '已配置' : '未配置'}</strong></div><div><ShieldCheck size={21} /><h3>语义重排</h3><strong>{health?.reranker ? '可用' : '未配置'}</strong></div><div><Gear size={21} /><h3>请求观测</h3><strong>{http?.requests || 0} 次</strong><small>失败 {http?.failures || 0} · 平均 {avgHttp} ms</small></div><div><Sparkle size={21} /><h3>LLM 观测</h3><strong>{llm?.calls || 0} 次</strong><small>Token {llm?.total_tokens || 0} · 平均 {avgLlm} ms</small></div><div><Database size={21} /><h3>向量后端</h3><strong>{metrics?.vector_backend || 'faiss'}</strong></div><div><ShieldCheck size={21} /><h3>访问控制</h3><strong>{metrics?.access_control ? '已启用' : '未启用'}</strong></div></div><div className="token-panel"><h3>访问令牌</h3><p>令牌仅保存在当前浏览器，用于可选令牌鉴权；后端未启用时不影响使用。</p><input type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="输入访问令牌" /><button className="secondary-button" onClick={saveToken} type="button">保存令牌</button></div></div>
}

export function App() {
  const [active, setActive] = useState('qa'), [library, setLibrary] = useState(null), [health, setHealth] = useState(null), [expert, setExpert] = useState(false)
  const [storageWarning, setStorageWarning] = useState('')
  const [history, setHistory] = useState(() => JSON.parse(localStorage.getItem('fire-review-history') || '[]'))
  const legacyTurns = readStoredJson('fire-review-qa-conversation', [])
  const storedSessions = readStoredJson('fire-review-sessions', null)
  const [sessions, setSessions] = useState(() => {
    if (Array.isArray(storedSessions?.sessions) && storedSessions.sessions.length) return storedSessions.sessions
    const first = legacyTurns.length ? { ...makeSession('历史问答'), qaTurns: legacyTurns.slice(-30) } : makeSession()
    return [first]
  })
  const [activeSessionId, setActiveSessionId] = useState(() => storedSessions?.activeSessionId || sessions[0].id)
  useEffect(() => { api.library().then(setLibrary).catch(() => setLibrary({ standards: [], total_chunks: 0 })); api.health().then(setHealth).catch(() => setHealth({ status: 'offline' })) }, [])
  useEffect(() => {
    const saved = writeStoredJson('fire-review-sessions', { sessions: sessions.slice(-50), activeSessionId })
    if (!saved) setStorageWarning('当前会话保存在本地浏览器失败，刷新后可能丢失。')
  }, [sessions, activeSessionId])
  useEffect(() => {
    const saved = writeStoredJson('fire-review-history', history)
    if (!saved) setStorageWarning('当前会话保存在本地浏览器失败，刷新后可能丢失。')
  }, [history])
  const addHistory = (item) => setHistory((current) => [{ ...item, at: new Date().toISOString() }, ...current].slice(0, 30))
  const updateSession = (id, patch) => setSessions((current) => current.map((session) => {
    if (session.id !== id) return session
    const next = typeof patch === 'function' ? patch(session) : { ...session, ...patch }
    next.updatedAt = new Date().toISOString()
    const titleCandidates = [
      (next.qaTurns || [])[0]?.question,
      (next.modelAnswers || [])[0]?.question,
      (next.reviewAnswers || [])[0]?.question,
      next.ifcFileName ? `IFC：${next.ifcFileName}` : '',
    ].filter(Boolean)
    if (titleCandidates.length && (!next.title || next.title === '新会话')) {
      const candidate = titleCandidates[0]
      next.title = candidate.length > 18 ? `${candidate.slice(0, 18)}…` : candidate
    }
    return next
  }))
  const createSession = () => {
    const session = makeSession()
    setSessions((current) => [...current, session])
    setActiveSessionId(session.id)
  }
  const selectSession = (id) => setActiveSessionId(id)
  const renameSession = (id) => {
    const session = sessions.find((item) => item.id === id)
    const title = window.prompt('会话名称', session?.title || '')
    if (title?.trim()) updateSession(id, { title: title.trim() })
  }
  const deleteSession = (id) => {
    const session = sessions.find((item) => item.id === id)
    if (!window.confirm(`删除会话“${session?.title || '未命名'}”？`)) return
    const remaining = sessions.filter((item) => item.id !== id)
    if (!remaining.length) {
      const fresh = makeSession()
      setSessions([fresh])
      setActiveSessionId(fresh.id)
      return
    }
    setSessions(remaining)
    if (id === activeSessionId) setActiveSessionId(remaining[0].id)
  }
  const activeSession = sessions.find((item) => item.id === activeSessionId) || sessions[0]
  return <AppShell active={active} setActive={setActive} library={library} health={health} expert={expert} setExpert={setExpert} sessions={sessions} activeSessionId={activeSession?.id} onSelectSession={selectSession} onCreateSession={createSession} onRenameSession={renameSession} onDeleteSession={deleteSession} storageWarning={storageWarning}>
    <div className={`workspace-view ${active === 'qa' ? 'active' : ''}`}><QAWorkspace library={library} expert={expert} onHistory={addHistory} session={activeSession} onUpdateSession={updateSession} /></div>
    <div className={`workspace-view ${active === 'bim' ? 'active' : ''}`}><BIMWorkspace library={library} expert={expert} onHistory={addHistory} session={activeSession} onUpdateSession={updateSession} sessions={sessions} activeSessionId={activeSession?.id} onCreateSession={createSession} onSelectSession={selectSession} /></div>
    <div className={`workspace-view ${active === 'versions' ? 'active' : ''}`}><VersionWorkspace /></div>
    <div className={`workspace-view ${active === 'library' ? 'active' : ''}`}><LibraryWorkspace library={library} /></div>
    <div className={`workspace-view ${active === 'history' ? 'active' : ''}`}><HistoryWorkspace history={history} clearHistory={() => { setHistory([]); localStorage.removeItem('fire-review-history') }} /></div>
    <div className={`workspace-view ${active === 'status' ? 'active' : ''}`}><StatusWorkspace health={health} library={library} /></div>
  </AppShell>
}
