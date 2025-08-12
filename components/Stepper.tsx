interface StepperProps {
  currentStep: number
}

export default function Stepper({ currentStep }: StepperProps) {
  const steps = [
    { number: 1, title: '画像を選ぶ' },
    { number: 2, title: '背景を選ぶ' },
    { number: 3, title: '最終確認' },
    { number: 4, title: '保存' }
  ]

  return (
    <div className="space-y-4">
      <h2 className="font-semibold text-sm text-gray-600 uppercase tracking-wide">進行状況</h2>
      {steps.map((step) => (
        <div key={step.number} className="flex items-center gap-3">
          <div className={`
            flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium
            ${step.number === currentStep 
              ? 'bg-brand-500 text-white' 
              : step.number < currentStep 
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-500'
            }
          `}>
            {step.number < currentStep ? '✓' : step.number}
          </div>
          <span className={`text-sm ${step.number === currentStep ? 'font-medium' : 'text-gray-500'}`}>
            {step.title}
          </span>
        </div>
      ))}
    </div>
  )
}