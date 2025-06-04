import discord
from discord.ext import commands
from discord import app_commands
import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document 

logger = logging.getLogger(__name__)

class RAG(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.qa_chain = None
        self.vector_store = None 
        self.retriever = None
        logger.info("RAG Cog initialized.")

    async def setup_hook(self):
        logger.info("RAG.setup_hook() called.")
        await self.initialize_rag()
        logger.info("RAG.setup_hook() completed.")

    async def initialize_rag(self):
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in .env")
            
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            texts = [
                "LFG offers services in web development.",
                "LFG provides solutions for cloud computing.",
                "LFG specializes in AI and Machine Learning applications.",
                "Contact LFG for a consultation on your next project."
            ]
            documents = [Document(page_content=t, metadata={"source": f"dummy_doc_{i}"}) for i, t in enumerate(texts)]

            self.vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
            
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2} 
            )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            logger.info("RAG system initialized successfully with FAISS")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}", exc_info=True) 
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
            result = self.qa_chain.invoke({"query": question}) 
            
            response = f"**Q:** {question}\n\n**A:** {result['result']}"
            if 'source_documents' in result and result['source_documents']:
                sources = set()
                for doc in result['source_documents']:
                    if hasattr(doc, 'metadata') and doc.metadata and 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                    else:
                        sources.add('Unknown source') 
                
                if sources:
                    sources_text = "\n\n*Sources:*\n" + "\n".join(f"- {src}" for src in sources)
                    response += sources_text
            
            await interaction.followup.send(response[:2000]) 
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            await interaction.followup.send("Sorry, I encountered an error while processing your question. Please try again later.", ephemeral=True)

    @app_commands.command(name="refresh_rag", description="Re-initializes the RAG system.")
    async def refresh_rag(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            logger.info("RAG refresh command invoked. Re-initializing RAG system...")
            await self.initialize_rag() 
            await interaction.followup.send("RAG system has been refreshed successfully with FAISS.", ephemeral=True)
            logger.info("RAG system refreshed successfully with FAISS.")
        except Exception as e:
            logger.error(f"Error during RAG refresh: {e}", exc_info=True)
            await interaction.followup.send(f"An error occurred during RAG refresh: {e}", ephemeral=True)

async def setup(bot):
    cog = RAG(bot)
    await bot.add_cog(cog)
    logger.info("RAG cog added to bot. Attempting to call cog.setup_hook() explicitly.")
    try:
        await cog.setup_hook() 
        logger.info("RAG cog loaded and cog.setup_hook() explicitly completed.")
    except Exception as e:
        logger.error(f"Error during RAG cog setup_hook: {e}", exc_info=True)
        raise
