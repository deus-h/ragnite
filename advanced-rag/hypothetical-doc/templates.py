"""
Domain-Specific HyDE Templates for RAGNITE

This module provides templates for generating hypothetical documents
across different domains and applications.
"""

from typing import Dict, List, Optional, Any

class HyDETemplates:
    """
    Collection of domain-specific HyDE templates to enhance hypothetical
    document generation for different use cases.
    """
    
    @staticmethod
    def get_domain_templates(domain: str = "general") -> Dict[str, Any]:
        """
        Get domain-specific templates for a particular domain.
        
        Args:
            domain: The domain to get templates for.
                Options: "general", "code", "medical", "legal", "scientific",
                "financial", "educational", "technical"
                
        Returns:
            Dictionary with domain-specific templates and settings
        """
        # Get template function based on domain
        template_functions = {
            "general": HyDETemplates.general_templates,
            "code": HyDETemplates.code_templates,
            "medical": HyDETemplates.medical_templates,
            "legal": HyDETemplates.legal_templates,
            "scientific": HyDETemplates.scientific_templates,
            "financial": HyDETemplates.financial_templates,
            "educational": HyDETemplates.educational_templates,
            "technical": HyDETemplates.technical_templates,
        }
        
        # Default to general templates if domain not found
        template_func = template_functions.get(domain, HyDETemplates.general_templates)
        return template_func()
    
    @staticmethod
    def general_templates() -> Dict[str, Any]:
        """General domain templates suitable for a wide range of queries."""
        
        # System prompt for general domain
        system_prompt = """
        You are an expert at generating hypothetical documents that would answer a query.
        Your goal is to create realistic, informative, and detailed content that contains
        the information someone would need to answer the query.
        
        Generate a document that:
        1. Directly addresses the topic of the query
        2. Contains factual information relevant to answering the query
        3. Includes specific details, examples, and explanations
        4. Is written in a clear, authoritative style
        5. Follows a logical structure with clear organization
        
        Do not write a direct answer to the query. Instead, create a document that would
        contain the information needed to formulate an answer.
        """
        
        # Perspective prompts for general domain
        perspective_prompts = {
            "informative": """
            Generate a hypothetical informational document that would contain information 
            needed to answer this query. The document should be comprehensive, factual, 
            and provide a thorough overview of the topic.
            
            Query: {query}
            
            Informational Document:
            """,
            
            "explanatory": """
            Generate a hypothetical explanatory document that would help answer this query.
            The document should explain concepts clearly, provide context, and help the
            reader understand the topic in depth.
            
            Query: {query}
            
            Explanatory Document:
            """,
            
            "comparative": """
            Generate a hypothetical document that compares and contrasts different aspects
            related to this query. The document should present multiple perspectives, options,
            or approaches, analyzing their similarities and differences.
            
            Query: {query}
            
            Comparative Document:
            """,
            
            "historical": """
            Generate a hypothetical document that provides historical context and background
            information relevant to this query. The document should trace developments over
            time and explain how the current situation or understanding evolved.
            
            Query: {query}
            
            Historical Document:
            """
        }
        
        # Recommended settings for general domain
        settings = {
            "num_perspectives": 3,
            "refinement_steps": 1,
            "temperature": 0.7
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def code_templates() -> Dict[str, Any]:
        """Code domain templates for programming and software development queries."""
        
        # System prompt for code domain
        system_prompt = """
        You are an expert software developer and technical writer. Your task is to generate
        hypothetical documentation that contains the information needed to answer a coding
        or software development query.
        
        Generate documentation that:
        1. Contains relevant code examples, patterns, or techniques
        2. Explains implementation details clearly
        3. Follows standard documentation practices for the relevant language/framework
        4. Includes both conceptual explanations and practical examples
        5. Addresses common pitfalls or edge cases
        
        Focus on creating content that a developer would find in high-quality documentation,
        tutorials, or technical blogs.
        """
        
        # Perspective prompts for code domain
        perspective_prompts = {
            "api_documentation": """
            Generate hypothetical API documentation that would contain information to answer
            this coding query. Include function signatures, parameter descriptions, return
            values, examples, and usage notes following standard documentation practices.
            
            Query: {query}
            
            API Documentation:
            """,
            
            "tutorial": """
            Generate a hypothetical tutorial that would explain how to implement or solve
            the issue in this query. Include step-by-step instructions, code examples with
            comments, and explanations of the approach.
            
            Query: {query}
            
            Tutorial:
            """,
            
            "technical_blog": """
            Generate a hypothetical technical blog post that would address this coding query.
            The post should provide in-depth explanations, best practices, and demonstrate
            expert knowledge about the topic with practical code examples.
            
            Query: {query}
            
            Technical Blog Post:
            """,
            
            "troubleshooting_guide": """
            Generate a hypothetical troubleshooting guide that would help solve the issue
            in this query. Include common problems, diagnostic steps, solutions with code
            examples, and explanations for why the solutions work.
            
            Query: {query}
            
            Troubleshooting Guide:
            """
        }
        
        # Recommended settings for code domain
        settings = {
            "num_perspectives": 2,
            "refinement_steps": 1,
            "temperature": 0.6
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def medical_templates() -> Dict[str, Any]:
        """Medical domain templates for healthcare and medical queries."""
        
        # System prompt for medical domain
        system_prompt = """
        You are an expert in medical science with extensive clinical and research experience.
        Your task is to generate hypothetical medical documents that would contain information
        to answer a healthcare or medical query.
        
        Generate documents that:
        1. Contain accurate medical information based on current evidence
        2. Follow standard medical terminology and formatting
        3. Include relevant clinical guidelines, research findings, or best practices
        4. Are appropriately detailed and precise
        5. Maintain a professional, clinical tone
        
        Remember to structure content as it would appear in medical literature, clinical
        guidelines, or educational resources for healthcare professionals.
        """
        
        # Perspective prompts for medical domain
        perspective_prompts = {
            "clinical_guidelines": """
            Generate hypothetical clinical guidelines that would contain information to answer
            this medical query. Include diagnostic criteria, management recommendations,
            treatment algorithms, and evidence quality assessments.
            
            Query: {query}
            
            Clinical Guidelines:
            """,
            
            "medical_textbook": """
            Generate a hypothetical medical textbook excerpt that would contain information
            to answer this medical query. Include pathophysiology, clinical presentations,
            diagnostic approaches, and management strategies with supporting evidence.
            
            Query: {query}
            
            Medical Textbook Excerpt:
            """,
            
            "research_review": """
            Generate a hypothetical systematic review or meta-analysis excerpt that would
            contain information to answer this medical query. Include study summaries,
            statistical findings, synthesis of evidence, and clinical implications.
            
            Query: {query}
            
            Research Review:
            """,
            
            "case_report": """
            Generate a hypothetical case report that would illustrate information relevant
            to this medical query. Include patient presentation, diagnostic workup, treatment
            approach, outcome, and discussion with references to literature.
            
            Query: {query}
            
            Case Report:
            """
        }
        
        # Recommended settings for medical domain
        settings = {
            "num_perspectives": 3,
            "refinement_steps": 2,
            "temperature": 0.5
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def legal_templates() -> Dict[str, Any]:
        """Legal domain templates for law and regulatory queries."""
        
        # System prompt for legal domain
        system_prompt = """
        You are a legal expert with extensive knowledge across multiple areas of law.
        Your task is to generate hypothetical legal documents that would contain information
        to answer a legal query.
        
        Generate documents that:
        1. Contain accurate legal information and principles
        2. Follow standard legal terminology, citation formats, and document structures
        3. Reference relevant statutes, cases, or regulations
        4. Consider jurisdictional factors where appropriate
        5. Maintain a formal, precise legal writing style
        
        Structure content as it would appear in legal memoranda, case analyses, legal treatises,
        or regulatory guidance.
        """
        
        # Perspective prompts for legal domain
        perspective_prompts = {
            "legal_memorandum": """
            Generate a hypothetical legal memorandum that would address this legal query.
            Include issue identification, relevant facts, legal analysis with case citations,
            statutory references, and a conclusion or recommendation.
            
            Query: {query}
            
            Legal Memorandum:
            """,
            
            "case_analysis": """
            Generate a hypothetical case analysis that would contain information relevant
            to this legal query. Include case facts, procedural history, legal issues,
            court's reasoning, holding, and implications with proper citations.
            
            Query: {query}
            
            Case Analysis:
            """,
            
            "legal_treatise": """
            Generate a hypothetical excerpt from a legal treatise that would provide
            information to answer this legal query. Include doctrinal explanation,
            historical development, current state of the law, and scholarly analysis.
            
            Query: {query}
            
            Legal Treatise Excerpt:
            """,
            
            "regulatory_guidance": """
            Generate hypothetical regulatory guidance that would address this legal query.
            Include interpretation of regulations, compliance requirements, examples of
            application, and best practices for regulatory compliance.
            
            Query: {query}
            
            Regulatory Guidance:
            """
        }
        
        # Recommended settings for legal domain
        settings = {
            "num_perspectives": 2,
            "refinement_steps": 2,
            "temperature": 0.4
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def scientific_templates() -> Dict[str, Any]:
        """Scientific domain templates for research and academic queries."""
        
        # System prompt for scientific domain
        system_prompt = """
        You are a scientific expert with extensive research and academic experience.
        Your task is to generate hypothetical scientific documents that would contain
        information to answer a scientific or research query.
        
        Generate documents that:
        1. Contain accurate scientific information based on current knowledge
        2. Follow standard scientific terminology and formatting
        3. Reference experimental evidence, methodologies, and theoretical frameworks
        4. Maintain objectivity and precision
        5. Follow scientific writing conventions
        
        Structure content as it would appear in research papers, review articles,
        methodology descriptions, or academic textbooks.
        """
        
        # Perspective prompts for scientific domain
        perspective_prompts = {
            "research_paper": """
            Generate a hypothetical research paper excerpt that would contain information
            to answer this scientific query. Include introduction, methods, results, and
            discussion sections with appropriate data presentation and analysis.
            
            Query: {query}
            
            Research Paper Excerpt:
            """,
            
            "review_article": """
            Generate a hypothetical scientific review article excerpt that would address
            this scientific query. Include comprehensive literature synthesis, current state
            of knowledge, theoretical frameworks, and future research directions.
            
            Query: {query}
            
            Review Article Excerpt:
            """,
            
            "methodology_description": """
            Generate a hypothetical methodology description that would contain information
            relevant to this scientific query. Include detailed protocols, experimental design,
            analytical techniques, controls, variables, and validation approaches.
            
            Query: {query}
            
            Methodology Description:
            """,
            
            "textbook_chapter": """
            Generate a hypothetical scientific textbook chapter excerpt that would explain
            concepts related to this scientific query. Include foundational principles,
            theoretical background, practical applications, and illustrative examples.
            
            Query: {query}
            
            Textbook Chapter Excerpt:
            """
        }
        
        # Recommended settings for scientific domain
        settings = {
            "num_perspectives": 3,
            "refinement_steps": 2,
            "temperature": 0.5
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def financial_templates() -> Dict[str, Any]:
        """Financial domain templates for business and finance queries."""
        
        # System prompt for financial domain
        system_prompt = """
        You are a financial expert with extensive experience in business, economics, and
        investment. Your task is to generate hypothetical financial documents that would
        contain information to answer a finance or business query.
        
        Generate documents that:
        1. Contain accurate financial information and analysis
        2. Use appropriate financial terminology and conventions
        3. Include relevant data, metrics, and calculations where appropriate
        4. Consider market conditions, regulatory factors, and business principles
        5. Maintain a professional analytical tone
        
        Structure content as it would appear in financial reports, investment analyses,
        market research, or business plans.
        """
        
        # Perspective prompts for financial domain
        perspective_prompts = {
            "financial_analysis": """
            Generate a hypothetical financial analysis that would address this finance query.
            Include quantitative data, ratio analysis, trend evaluation, comparative benchmarks,
            and actionable insights with appropriate financial metrics.
            
            Query: {query}
            
            Financial Analysis:
            """,
            
            "market_report": """
            Generate a hypothetical market report that would contain information relevant to
            this business or finance query. Include market trends, competitive landscape,
            growth projections, risk factors, and strategic implications.
            
            Query: {query}
            
            Market Report:
            """,
            
            "investment_memo": """
            Generate a hypothetical investment memorandum that would provide information to
            answer this financial query. Include investment thesis, risk assessment, valuation
            analysis, return projections, and strategic considerations.
            
            Query: {query}
            
            Investment Memorandum:
            """,
            
            "economic_outlook": """
            Generate a hypothetical economic outlook document that would address this finance
            or business query. Include macroeconomic indicators, forecasts, policy implications,
            sector impacts, and scenario analyses.
            
            Query: {query}
            
            Economic Outlook:
            """
        }
        
        # Recommended settings for financial domain
        settings = {
            "num_perspectives": 2,
            "refinement_steps": 1,
            "temperature": 0.6
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def educational_templates() -> Dict[str, Any]:
        """Educational domain templates for teaching and learning queries."""
        
        # System prompt for educational domain
        system_prompt = """
        You are an educational expert with extensive teaching and curriculum development
        experience. Your task is to generate hypothetical educational documents that would
        contain information to answer a teaching or learning query.
        
        Generate documents that:
        1. Present information in clear, structured ways suitable for learning
        2. Include examples, explanations, and practical applications
        3. Address different learning styles and levels
        4. Build from foundational to advanced concepts
        5. Incorporate educational best practices
        
        Structure content as it would appear in lesson plans, textbooks, educational guides,
        or instructional materials.
        """
        
        # Perspective prompts for educational domain
        perspective_prompts = {
            "lesson_plan": """
            Generate a hypothetical lesson plan that would address this educational query.
            Include learning objectives, instructional activities, assessments, materials
            needed, and differentiation strategies for various learners.
            
            Query: {query}
            
            Lesson Plan:
            """,
            
            "textbook_excerpt": """
            Generate a hypothetical textbook excerpt that would contain information to
            answer this educational query. Include conceptual explanations, examples,
            visual aids, practice problems, and connections to real-world applications.
            
            Query: {query}
            
            Textbook Excerpt:
            """,
            
            "study_guide": """
            Generate a hypothetical study guide that would help answer this educational
            query. Include key concepts, definitions, summaries, practice questions,
            mnemonics, and learning strategies to master the material.
            
            Query: {query}
            
            Study Guide:
            """,
            
            "teaching_resource": """
            Generate a hypothetical teaching resource that would provide information
            relevant to this educational query. Include instructional strategies,
            classroom activities, assessment tools, and supplementary materials.
            
            Query: {query}
            
            Teaching Resource:
            """
        }
        
        # Recommended settings for educational domain
        settings = {
            "num_perspectives": 3,
            "refinement_steps": 1,
            "temperature": 0.7
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        }
    
    @staticmethod
    def technical_templates() -> Dict[str, Any]:
        """Technical domain templates for engineering and technical queries."""
        
        # System prompt for technical domain
        system_prompt = """
        You are a technical expert with extensive engineering and technology experience.
        Your task is to generate hypothetical technical documents that would contain
        information to answer an engineering or technical query.
        
        Generate documents that:
        1. Contain accurate technical information and specifications
        2. Follow industry-standard terminology and conventions
        3. Include diagrams, data, or specifications where appropriate
        4. Address practical implementation considerations
        5. Maintain a precise, clear technical writing style
        
        Structure content as it would appear in technical specifications, engineering
        reports, user manuals, or technical documentation.
        """
        
        # Perspective prompts for technical domain
        perspective_prompts = {
            "technical_specification": """
            Generate a hypothetical technical specification that would address this
            engineering or technical query. Include requirements, functional specifications,
            performance parameters, standards compliance, and implementation guidelines.
            
            Query: {query}
            
            Technical Specification:
            """,
            
            "engineering_report": """
            Generate a hypothetical engineering report that would contain information
            relevant to this technical query. Include problem definition, methodology,
            analysis, findings, and recommendations with supporting data and calculations.
            
            Query: {query}
            
            Engineering Report:
            """,
            
            "user_manual": """
            Generate a hypothetical user manual excerpt that would provide information
            to answer this technical query. Include installation procedures, configuration
            steps, operation instructions, troubleshooting guidance, and technical specifications.
            
            Query: {query}
            
            User Manual Excerpt:
            """,
            
            "technical_whitepaper": """
            Generate a hypothetical technical whitepaper that would address this engineering
            or technical query. Include technology overview, architectural design, technical
            approach, performance analysis, and implementation considerations.
            
            Query: {query}
            
            Technical Whitepaper:
            """
        }
        
        # Recommended settings for technical domain
        settings = {
            "num_perspectives": 2,
            "refinement_steps": 1,
            "temperature": 0.6
        }
        
        return {
            "system_prompt": system_prompt,
            "perspective_prompts": perspective_prompts,
            "settings": settings
        } 