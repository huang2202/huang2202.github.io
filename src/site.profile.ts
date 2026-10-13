// Writing remains private until the author explicitly chooses to publish it.
export const publication = { writing: false }

const affiliationLinks = [
  { name: 'Zhejiang University', href: 'https://www.zju.edu.cn/english/' },
  { name: 'State Key Lab of CAD & CG', href: 'http://www.cad.zju.edu.cn/' }
]

const biography: { text: string; href?: string }[][] = [
  [
    { text: 'I am a first-year Ph.D. student in the direct-entry program at the ' },
    {
      text: 'College of Computer Science and Technology',
      href: 'https://www.zju.edu.cn/english/74913/list.psp'
    },
    { text: ', Zhejiang University. I am jointly trained at ' },
    { text: 'Zhongguancun Academy', href: 'https://www.bza.edu.cn/en/' },
    { text: ' in Beijing.' }
  ],
  [
    { text: 'I am advised by Prof. ' },
    { text: 'Chunhua Shen', href: 'https://cshen.github.io/' },
    {
      text: ' and am a member of his group at the State Key Lab of CAD & CG. I have also benefited greatly from '
    },
    { text: 'Hao Chen', href: 'https://stan-haochen.github.io/' },
    { text: '’s research guidance.' }
  ]
]

export const profile = {
  name: 'Guangyu Huang',
  alias: 'Harry Guang',
  affiliation: affiliationLinks.map(({ name }) => name).join(', '),
  affiliationLinks,
  biography,
  email: 'zju_hgy@163.com',
  github: 'https://github.com/huang2202',
  summary:
    'My research interests lie in robot learning and embodied intelligence, with a focus on reinforcement learning.',
  interests: [
    {
      title: 'Robot Learning & Embodied Intelligence',
      description:
        'My interests center on how robots can generalize learned skills to new objects, tasks, and environments, and adapt their behavior when conditions change.'
    },
    {
      title: 'Reinforcement Learning',
      description:
        'I am interested in combining demonstration data with reinforcement learning to acquire and refine skills, with an emphasis on making efficient use of limited interaction with the environment.'
    }
  ]
}
