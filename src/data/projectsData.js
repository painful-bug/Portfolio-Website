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
      'Enterprise data infrastructure often suffers from fragmented sources and inconsistent data quality. This project tackles the challenge of unifying heterogeneous CRM and ERP system data into a single analytics-ready source of truth using Microsoft SQL Server and production-grade Medallion Architecture.',
      'The three-tier pipeline ingests raw data via BULK INSERT into Bronze tables, applies cleansing and deduplication in Silver (using ROW_NUMBER window functions, CASE-based standardization, and NULL handling), then materializes a star schema in Gold with dim_customers, dim_products, and fact_sales views—complete with SCD Type 2 support and surrogate key generation.',
    ],
    technicalApproach: 'Built idempotent ETL stored procedures with TRY-CATCH error handling, execution-time logging, and TABLOCK-optimized bulk loading. Silver layer transformations include TRIM-based cleansing, categorical standardization (e.g., M→Male, S→Single), date format conversion, and audit timestamps for data lineage tracking across all layers.',
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
      'Driver distraction is a leading cause of road accidents. This project developed a deep learning system that classifies 10 distinct driver behaviors from the State Farm Kaggle dataset, processing 22,384 image sequences using a sliding window approach (stride=1, sequence length=5) for temporal context.',
      'The custom YOLO_LSTM_Attn architecture uses a frozen YOLOv8n backbone wrapped in a YOLOBackboneWrapper that extracts 256-dimensional feature vectors through SPPF-layer truncation and adaptive average pooling, feeding them into a bidirectional LSTM with a learned TemporalAttention mechanism for sequence-aware classification.',
    ],
    technicalApproach: 'Implemented a memory-efficient pipeline for 4GB VRAM: frozen YOLOv8n backbone (layers 0-9) produces [B,256,20,20] feature maps, pooled to 256-dim vectors per frame. WeightedRandomSampler handles class imbalance, while mixed-precision training (torch.cuda.amp.GradScaler) enables batch_size=2 training. The TemporalAttention module learns soft attention scores over LSTM hidden states for context-weighted prediction.',
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
    tags: ['Python', 'FastAPI'],
    role: 'AI Developer',
    year: '2024',
    techStack: ['Python', 'FastAPI', 'GPT-4o-mini', 'Docker'],
    description: [
      'A Dockerized FastAPI-based AI automation platform that intelligently routes natural-language task descriptions to either a library of predefined functions or an autonomous code-generation agent powered by GPT-4o-mini. It handles a diverse range of tasks including data formatting, contact sorting, log analysis, Markdown indexing, email parsing, credit card OCR, semantic similarity via embeddings, SQL queries, API data fetching, git operations, and web scraping.',
      'For tasks outside the predefined library, the AIAgent class implements an autonomous generate-execute-debug loop: it dynamically produces Python code, installs required packages, executes in a subprocess sandbox, and self-corrects through up to 3 iterative rounds of error analysis and code regeneration—all without human intervention.',
    ],
    technicalApproach: 'The FastAPI endpoint receives task descriptions, then a helper function uses LLM-based intent classification to match against known function signatures in the tasks module. Unmatched requests are forwarded to the AIAgent, which generates system prompts with strict workflow instructions, extracts code blocks via regex, manages package installation, and maintains conversation history for contextual multi-turn debugging.',
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
    tags: ['LangChain', 'RAG'],
    role: 'AI Engineer',
    year: '2024',
    techStack: ['Python', 'FastAPI', 'LangChain', 'React', 'Google Gemini', 'Firebase'],
    description: [
      'A comprehensive AI-powered financial assistant built for the GDG Solution Challenge 2025, featuring a React + Material-UI frontend with Firebase authentication and a Python backend powered by LangChain, the Agno framework, and Google Gemini API. Users ask natural-language questions about stocks, market trends, and investment strategies.',
      'The system combines a RAG pipeline (FAISS/Chroma vector store with relevance grading) for domain knowledge retrieval with a Deep Research Engine that recursively decomposes complex queries into sub-questions, processes them in parallel via Tavily web search and YFinance real-time data lookups, and synthesizes findings into coherent, evidence-backed responses.',
    ],
    technicalApproach: 'The processing pipeline routes queries through small-talk detection, RAG-based knowledge retrieval with relevance scoring, optional deep research mode (recursive query decomposition with configurable depth), real-time financial data integration via YFinance API, and Tavily-powered web search—all orchestrated by LangChain agent workflows with persistent conversational memory and session management.',
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
