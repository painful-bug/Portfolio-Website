const projects = [
  {
    id: 1,
    title: 'Data Warehouse',
    subtitle: 'Star Schema & Medallion Architecture for production-grade analytics',
    image: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&q=80',
    heroImage: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80',
    detailImage: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&q=80',
    tags: ['MS SQL Server', 'ETL'],
    role: 'Data Engineer & Architect',
    year: '2024',
    techStack: ['MS SQL Server', 'ETL Pipelines', 'Star Schema', 'Medallion Architecture'],
    description: [
      'Enterprise data infrastructure often suffers from fragmented sources and inconsistent data quality. This project tackled the challenge of unifying CRM and ERP data sources with 18,500+ customer records into a single source of truth using production-grade patterns.',
      'By implementing the Medallion Architecture (Bronze-Silver-Gold layers), we established a clear data lineage from raw ingestion to analytics-ready dimensions. The star schema design with SCD Type 2 support enables historical tracking without sacrificing query performance.',
    ],
    technicalApproach: 'Built automated ETL pipelines using stored procedures with comprehensive data cleansing, deduplication via window functions (ROW_NUMBER, RANK), and robust error handling. The dimensional model features fact and dimension tables optimized for Tableau dashboards and ad-hoc SQL analysis.',
    codeSnippet: `-- Medallion Architecture: Bronze → Silver → Gold
CREATE PROCEDURE etl.load_silver_customers AS
BEGIN
  WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY customer_id
      ORDER BY load_date DESC
    ) AS rn FROM bronze.customers
  )
  MERGE silver.customers AS target
  USING (SELECT * FROM ranked WHERE rn = 1) AS source
  ON target.customer_id = source.customer_id
  WHEN MATCHED THEN UPDATE SET ...
  WHEN NOT MATCHED THEN INSERT ...;
END;`,
    span: 'full',
    articleTitle: 'Engineering Truth from Chaos',
  },
  {
    id: 2,
    title: 'Driver Distraction Detection',
    subtitle: 'Hybrid CNN-LSTM architecture for real-time driver behavior classification',
    image: 'https://images.unsplash.com/photo-1549317661-bd32c8ce0aca?w=800&q=80',
    heroImage: 'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=1200&q=80',
    detailImage: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&q=80',
    tags: ['PyTorch', 'YOLOv8'],
    role: 'ML Engineer & Researcher',
    year: '2024',
    techStack: ['PyTorch', 'YOLOv8', 'LSTM', 'Computer Vision'],
    description: [
      'Driver distraction is a leading cause of road accidents. This project developed a deep learning system capable of classifying 10 distinct driver distraction behaviors from the State Farm Kaggle dataset containing 22,000+ images.',
      'The hybrid architecture combines the spatial feature extraction power of YOLOv8 with the temporal reasoning capabilities of LSTM networks, creating a system that understands not just what a driver is doing, but the temporal context of their actions.',
    ],
    technicalApproach: 'Implemented a sequence-based learning approach with sliding window for temporal context modeling. Used transfer learning with frozen YOLOv8 pretrained weights, weighted random sampling to handle class imbalance, and mixed precision training (FP16) for memory optimization on a 4GB GPU constraint.',
    codeSnippet: `class HybridCNNLSTM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = yolov8_backbone(pretrained=True)
        self.backbone.requires_grad_(False)
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=2, batch_first=True
        )
        self.attention = TemporalAttention(256)
        self.classifier = nn.Linear(256, num_classes)`,
    span: 'half',
    articleTitle: 'Seeing Beyond the Frame',
  },
  {
    id: 3,
    title: 'Agentic AI Application',
    subtitle: 'Autonomous AI system with dynamic code generation & self-correction',
    image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&q=80',
    heroImage: 'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&q=80',
    detailImage: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=800&q=80',
    tags: ['Python', 'Prompt Engineering'],
    role: 'AI Developer',
    year: '2024',
    techStack: ['Python', 'Prompt Chaining', 'Self-Prompting', 'Code Generation'],
    description: [
      'Traditional AI assistants are constrained by their training data and predefined capabilities. This project pushes beyond those boundaries by creating an autonomous application that can dynamically generate, validate, and execute Python code for novel requests it has never encountered.',
      'The system implements sophisticated prompt chaining and self-prompting techniques, creating a feedback loop where the AI can detect errors in its own generated code, correct them, and re-execute—all without human intervention.',
    ],
    technicalApproach: 'Built a multi-stage pipeline: intent classification → task decomposition → code generation → static analysis → sandboxed execution → error detection → self-correction loop. Each stage uses carefully engineered prompts that maintain context across the chain.',
    codeSnippet: `class AgenticPipeline:
    def execute(self, user_request):
        intent = self.classify_intent(user_request)
        if intent == "novel_task":
            code = self.generate_code(user_request)
            result = self.sandbox_execute(code)
            if result.has_error:
                code = self.self_correct(code, result.error)
                result = self.sandbox_execute(code)
        return result`,
    span: 'half',
    stagger: true,
    articleTitle: 'Teaching Machines to Think',
  },
  {
    id: 4,
    title: 'Financial Assistant AI',
    subtitle: 'Intelligent agent for web research & comprehensive financial analysis',
    image: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&q=80',
    heroImage: 'https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=1200&q=80',
    detailImage: 'https://images.unsplash.com/photo-1642790106117-e829e14a795f?w=800&q=80',
    tags: ['Web Research', 'NLP'],
    role: 'AI Engineer',
    year: '2024',
    techStack: ['Python', 'NLP', 'Web Scraping', 'Reasoning Algorithms'],
    description: [
      'Financial research requires synthesizing vast amounts of real-time data from diverse sources. This AI agent automates the entire research workflow—from intelligent web searching to deep analysis—producing comprehensive financial reports.',
      'The agent utilizes advanced reasoning algorithms to understand complex user queries, identify relevant data sources, and synthesize information into actionable insights rather than raw data dumps.',
    ],
    technicalApproach: 'The agent operates through a multi-step reasoning pipeline: query decomposition → targeted web searches → content extraction → fact verification → synthesis → report generation. Each step uses chain-of-thought reasoning to maintain analytical rigor.',
    codeSnippet: `class FinancialAgent:
    async def research(self, query):
        sub_queries = self.decompose(query)
        sources = await asyncio.gather(*[
            self.search_web(q) for q in sub_queries
        ])
        facts = self.extract_and_verify(sources)
        report = self.synthesize(facts, query)
        return report`,
    span: 'half',
    articleTitle: 'Intelligence at Scale',
  },
  {
    id: 5,
    title: 'Human Action Recognition',
    subtitle: '3D CNN model for classifying human actions from video data',
    image: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&q=80',
    heroImage: 'https://images.unsplash.com/photo-1535378917042-10a22c95931a?w=1200&q=80',
    detailImage: 'https://images.unsplash.com/photo-1507146426996-ef05306b995a?w=800&q=80',
    tags: ['TensorFlow', 'Deep Learning'],
    role: 'ML Researcher',
    year: '2024',
    techStack: ['TensorFlow', 'Computer Vision', '3D CNNs', 'Video Processing'],
    description: [
      'Understanding human actions from video requires reasoning across both spatial and temporal dimensions simultaneously. Traditional 2D CNNs lose critical temporal information by processing individual frames independently.',
      'This project leverages 3D convolutional neural networks that process video volumes directly, capturing motion patterns and temporal relationships that are invisible to frame-by-frame analysis.',
    ],
    technicalApproach: 'Built a 3D CNN architecture using TensorFlow that processes fixed-length video clips as volumetric inputs. The 3D convolution kernels span the temporal dimension, enabling the network to learn spatiotemporal features directly from raw video data.',
    codeSnippet: `model = tf.keras.Sequential([
    layers.Conv3D(64, (3, 3, 3), activation='relu',
                  input_shape=(16, 224, 224, 3)),
    layers.MaxPool3D((1, 2, 2)),
    layers.Conv3D(128, (3, 3, 3), activation='relu'),
    layers.MaxPool3D((2, 2, 2)),
    layers.GlobalAveragePooling3D(),
    layers.Dense(num_classes, activation='softmax')
])`,
    span: 'half',
    stagger: true,
    articleTitle: 'Motion as Language',
  },
];

export default projects;
