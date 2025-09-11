interface StepCardProps {
  stepNumber: number
  title: string
  children: React.ReactNode
}

export default function StepCard({ stepNumber, title, children }: StepCardProps) {
  return (
    <div className="card">
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