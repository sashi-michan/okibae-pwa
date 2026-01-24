import { forwardRef } from 'react'

interface StepCardProps {
  stepNumber: number
  title: string
  children: React.ReactNode
  className?: string
}

const StepCard = forwardRef<HTMLDivElement, StepCardProps>(
  ({ stepNumber, title, children, className }, ref) => {
    return (
      <div ref={ref} className={`card ${className || ''}`}>
        <div className="flex items-center gap-2 mb-4">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand-500 text-white text-xs font-medium">
            {stepNumber}
          </span>
          <h2 className="typography-step-title">{title}</h2>
        </div>
        {children}
      </div>
    )
  }
)

StepCard.displayName = 'StepCard'

export default StepCard