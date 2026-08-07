import { useState, useRef } from 'react'

const API = 'http://localhost:8000'

export default function App() {
  const [maze, setMaze] = useState(null)
  const [start, setStart] = useState(null)
  const [goal, setGoal] = useState(null)
  const [agentPos, setAgentPos] = useState(null)
  const [path, setPath] = useState([])
  const [status, setStatus] = useState('idle') // idle | loading | solving | done
  const [result, setResult] = useState(null)
  const timerRef = useRef(null)

  async function fetchNewMaze() {
    clearInterval(timerRef.current)
    setStatus('loading')
    setResult(null)
    setPath([])
    const res = await fetch(`${API}/new-maze`)
    const data = await res.json()
    setMaze(data.maze)
    setStart(data.start)
    setGoal(data.goal)
    setAgentPos(data.start)
    setStatus('idle')
  }

  async function solve() {
    if (!maze) return
    setStatus('solving')
    const res = await fetch(`${API}/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ maze, start, goal }),
    })
    const data = await res.json()
    setPath(data.path)
    animatePath(data.path, data)
  }

  function animatePath(fullPath, finalResult) {
    let i = 0
    timerRef.current = setInterval(() => {
      setAgentPos(fullPath[i])
      i += 1
      if (i >= fullPath.length) {
        clearInterval(timerRef.current)
        setStatus('done')
        setResult(finalResult)
      }
    }, 300)
  }

  return (
    <div className="page">
      <header className="header">
        <span className="eyebrow">PPO / gymnasium</span>
        <h1>Maze Agent</h1>
        <p className="sub">A reinforcement-learning agent trained to reach the goal in mazes it has never seen.</p>
      </header>

      <div className="controls">
        <button onClick={fetchNewMaze} disabled={status === 'solving'}>
          New maze
        </button>
        <button onClick={solve} disabled={!maze || status === 'solving'} className="primary">
          {status === 'solving' ? 'Solving…' : 'Solve'}
        </button>
      </div>

      {maze ? (
        <Grid maze={maze} start={start} goal={goal} agentPos={agentPos} />
      ) : (
        <div className="empty">Generate a maze to begin.</div>
      )}

      {result && (
        <div className={`result ${result.success ? 'ok' : 'fail'}`}>
          {result.success
            ? `Reached the goal in ${result.steps} steps.`
            : `Did not reach the goal within the step limit (${result.steps} steps taken).`}
        </div>
      )}
    </div>
  )
}

function Grid({ maze, start, goal, agentPos }) {
  const height = maze.length
  const width = maze[0].length

  return (
    <div
      className="grid"
      style={{ gridTemplateColumns: `repeat(${width}, 1fr)`, gridTemplateRows: `repeat(${height}, 1fr)` }}
    >
      {maze.map((row, r) =>
        row.map((cell, c) => {
          const isStart = start && start[0] === r && start[1] === c
          const isGoal = goal && goal[0] === r && goal[1] === c
          const isAgent = agentPos && agentPos[0] === r && agentPos[1] === c
          const isWall = cell === 1

          let className = 'cell'
          if (isWall) className += ' wall'
          if (isGoal) className += ' goal'
          if (isStart) className += ' start'

          return (
            <div key={`${r}-${c}`} className={className}>
              {isAgent && <div className="agent" />}
            </div>
          )
        })
      )}
    </div>
  )
}
