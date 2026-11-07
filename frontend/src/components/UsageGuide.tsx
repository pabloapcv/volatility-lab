import { useEffect, useState } from 'react'
import { ChevronDown, HelpCircle } from 'lucide-react'

const STORAGE_PREFIX = 'volatilitylab:howto:'

interface UsageGuideProps {
  storageKey: string
  items: string[]
}

export default function UsageGuide({ storageKey, items }: UsageGuideProps) {
  const lsKey = `${STORAGE_PREFIX}${storageKey}`
  const [open, setOpen] = useState(true)

  useEffect(() => {
    try {
      const saved = localStorage.getItem(lsKey)
      if (saved === 'collapsed') setOpen(false)
      if (saved === 'open') setOpen(true)
    } catch {
      // private mode / unavailable storage
    }
  }, [lsKey])

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev
      try {
        localStorage.setItem(lsKey, next ? 'open' : 'collapsed')
      } catch {
        // ignore
      }
      return next
    })
  }

  return (
    <div className="card">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-gray-900">
          <HelpCircle className="h-4 w-4 text-primary-600" />
          How to use this page
        </span>
        <ChevronDown
          className={`h-4 w-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open && (
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-gray-600">
          {items.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
