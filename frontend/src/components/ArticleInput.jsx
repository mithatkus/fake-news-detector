import { useState } from 'react'

const REAL_EXAMPLES = [
  `WASHINGTON (Reuters) - The White House said on Monday it was reviewing a proposed rule that would require U.S. businesses to verify the citizenship status of workers. The administration has been examining regulations on immigration enforcement across multiple agencies, according to administration officials familiar with the matter. The rule, if implemented, would expand use of the federal E-Verify system and could affect millions of employers nationwide. Business groups have raised concerns about the administrative burden and potential disruption to hiring processes.`,

  `WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday, signaling a cautious approach as policymakers weighed slowing inflation against persistent strength in the labor market. Fed Chair Jerome Powell said at a news conference that officials needed more data before cutting borrowing costs. The central bank's benchmark overnight interest rate remains in the 5.25%–5.50% range where it has been since last summer. Investors had widely expected no change, though markets are pricing in several cuts later in the year if inflation continues to ease toward the Fed's 2% target.`,

  `BRUSSELS (Reuters) - The European Union agreed on Thursday to impose additional tariffs on Chinese electric vehicles, escalating a trade dispute that has rattled global markets. The duties, ranging from 17% to 38%, will take effect next month. Chinese officials condemned the move as protectionist and threatened retaliatory measures targeting European luxury goods and agricultural products. The announcement sent shares of German automakers sharply lower as investors worried about the impact of a prolonged trade conflict on supply chains and export revenues across the continent.`,

  `WASHINGTON (Reuters) - U.S. health regulators on Friday approved a new Alzheimer's drug that clinical trials showed could slow the progression of the disease in early stages. The Food and Drug Administration granted full approval after a review panel found the treatment reduced cognitive decline by 27% in patients with mild Alzheimer's over an 18-month period. The approval marks a significant milestone in treating a disease that affects more than six million Americans. The drug will cost approximately $26,500 per year, raising concerns among patient advocacy groups about affordability and insurance coverage.`,

  `UNITED NATIONS (Reuters) - The United Nations Security Council voted on Tuesday to extend peacekeeping operations in a conflict-affected region for another six months, with 13 members in favor and two abstentions. The resolution authorizes up to 12,000 troops to remain deployed and calls for continued monitoring of a fragile ceasefire that ended months of heavy fighting. Humanitarian agencies said the extension was critical to maintaining access for aid deliveries to more than two million displaced civilians. Diplomats acknowledged the negotiations had been difficult, with disagreements over the scope of the mandate and the length of the renewal period.`,
]

const FAKE_EXAMPLES = [
  `BREAKING: Secret documents leaked from inside the Pentagon reveal a massive government conspiracy to hide evidence of alien contact. Anonymous sources confirm that top officials have been paid to suppress the truth for decades. Share this before it gets taken down! The mainstream media won't cover this story — we are the only outlet brave enough to report it. Photographs, flight logs, and internal memos allegedly prove that recovered spacecraft have been stored at classified facilities since the 1950s.`,

  `EXCLUSIVE: Whistleblower scientists working inside a top pharmaceutical laboratory have come forward with bombshell evidence that COVID-19 vaccines contain microscopic tracking devices linked to a global surveillance network. The nano-chips, allegedly developed with funding from a secretive billionaire foundation, are activated by 5G towers and can monitor location and biological data in real time. Dozens of doctors who attempted to speak out have reportedly had their medical licenses revoked. Independent researchers claim to have observed self-assembling structures under dark-field microscopy not listed in any official ingredient disclosures.`,

  `BOMBSHELL REPORT: A team of forensic auditors working in secret has uncovered definitive proof that millions of ballots were systematically destroyed and replaced in five battleground states. Sources within the intelligence community, speaking anonymously to protect their lives, say the fraud was orchestrated by foreign actors working with domestic operatives embedded inside voting machine companies. The evidence, reportedly stored on encrypted hard drives, shows timestamps inconsistent with official vote counts. Major networks are refusing to cover the story after receiving direct orders from their corporate owners to bury it permanently.`,

  `EXPOSED: Government documents obtained through a freedom of information request reveal that federal agencies have been conducting a secret atmospheric spraying program over major U.S. cities for over two decades. The chemical compounds — including barium, strontium, and aluminum oxide — are allegedly designed to suppress critical thinking and increase population docility. Former military pilots who participated in the program have broken their silence, describing orders to fly grid patterns over urban areas during early morning hours when residents are least likely to notice the trails being deliberately laid overhead.`,

  `URGENT: Leaked emails between senior government officials and a shadowy globalist organization prove that a coordinated effort is underway to dismantle national sovereignty and install a one-world government by the end of the decade. The documents, authenticated by a former intelligence contractor now living in exile, detail a 12-step plan to engineer economic crises and mass social unrest as pretexts for centralizing power. Tech billionaires named in the correspondence have already begun constructing underground facilities in remote locations, and this information has been aggressively suppressed across every major social media platform.`,
]

function pickRandom(pool, lastIdx) {
  if (pool.length === 1) return 0
  let idx
  do {
    idx = Math.floor(Math.random() * pool.length)
  } while (idx === lastIdx)
  return idx
}

export default function ArticleInput({ value, onChange }) {
  const [lastRealIdx, setLastRealIdx] = useState(-1)
  const [lastFakeIdx, setLastFakeIdx] = useState(-1)

  function loadReal() {
    const idx = pickRandom(REAL_EXAMPLES, lastRealIdx)
    setLastRealIdx(idx)
    onChange(REAL_EXAMPLES[idx])
  }

  function loadFake() {
    const idx = pickRandom(FAKE_EXAMPLES, lastFakeIdx)
    setLastFakeIdx(idx)
    onChange(FAKE_EXAMPLES[idx])
  }

  const wordCount = value.trim().split(/\s+/).filter(Boolean).length
  const isTooShort = value.trim().length > 0 && wordCount < 30

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={loadReal}
          className="text-sm px-3 py-1.5 rounded-lg border border-green-300 text-green-700 hover:bg-green-50 transition-colors"
        >
          Load Real Example
        </button>
        <button
          type="button"
          onClick={loadFake}
          className="text-sm px-3 py-1.5 rounded-lg border border-red-300 text-red-700 hover:bg-red-50 transition-colors"
        >
          Load Fake Example
        </button>
        {value && (
          <button
            type="button"
            onClick={() => onChange('')}
            className="text-sm px-3 py-1.5 rounded-lg border border-gray-300 text-gray-500 hover:bg-gray-50 transition-colors ml-auto"
          >
            Clear
          </button>
        )}
      </div>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={10}
        placeholder="Paste the full text of a news article here…"
        className="w-full rounded-xl border border-gray-300 p-4 text-sm leading-relaxed
                   focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent
                   resize-y placeholder-gray-400"
      />

      {isTooShort ? (
        <p className="text-sm text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Please enter a full news article for accurate results — both models were trained on article-length text.
        </p>
      ) : (
        <p className="text-xs text-gray-400 text-right">
          {wordCount} words
        </p>
      )}
    </div>
  )
}
