import os
import logging
import discord
from discord import app_commands
from discord.ext import commands
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client, Client
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class RAG(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.supabase: Optional[Client] = None
        self.initialize_supabase()
        self.initialize_rag()

    def initialize_supabase(self):
        """Initialize Supabase client."""
        try:
            url = os.getenv('SUPABASE_URL')
            key = os.getenv('SUPABASE_KEY')
            if not url or not key:
                raise ValueError("Supabase URL and key must be provided in .env")
            self.supabase = create_client(url, key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Supabase: {e}")
            raise

    def initialize_rag(self):
        """Initialize the RAG components with Supabase."""
        try:
            if not self.supabase:
                raise ValueError("Supabase client not initialized")

            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create Supabase vector store
            table_name = os.getenv('SUPABASE_TABLE', 'documents')
            
            # Initialize the vector store with Supabase
            self.vector_store = SupabaseVectorStore(
                embedding=embeddings,
                client=self.supabase,
                table_name=table_name,
                query_name=f"match_documents"
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Set up QA chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            logger.info("RAG system initialized successfully with Supabase")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    @app_commands.command(name="ask", description="Ask a question about LFG services")
    @app_commands.describe(question="Your question about LFG services")
    async def ask_question(self, interaction: discord.Interaction, question: str):
        """Ask a question about LFG services."""
        if not self.qa_chain:
            await interaction.response.send_message("The question-answering system is not ready yet. Please try again later.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True)

        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Format the response
            response = f"**Q:** {question}\n\n**A:** {result['result']}"
            
            # Add sources if available
            if 'source_documents' in result and result['source_documents']:
                sources = set(doc.metadata.get('source', 'Unknown') for doc in result['source_documents'])
                sources_text = "\n\n*Sources:*\n" + "\n".join(f"- {src}" for src in sources)
                response += sources_text
            
            # Send the response (Discord has a 2000 character limit per message)
            await interaction.followup.send(response[:2000])
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            await interaction.followup.send("Sorry, I encountered an error while processing your question. Please try again later.", ephemeral=True)

    @app_commands.command(name="refresh", description="Refresh the knowledge base (Admin only)")
    @app_commands.checks.has_permissions(administrator=True)
    async def refresh_knowledge(self, interaction: discord.Interaction):
        """Refresh the knowledge base (Admin only)."""
        await interaction.response.defer(ephemeral=True)
        try:
            await interaction.followup.send("Refreshing knowledge base. This may take a moment...", ephemeral=True)
            self.initialize_rag()
            await interaction.followup.send("Knowledge base refreshed successfully!", ephemeral=True)
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}", exc_info=True)
            await interaction.followup.send("Failed to refresh knowledge base. Check logs for details.", ephemeral=True)

async def setup(bot):
    await bot.add_cog(RAG(bot))
