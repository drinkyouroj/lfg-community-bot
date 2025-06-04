import os
import logging
import discord
from discord.ext import commands
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from typing import Optional

logger = logging.getLogger(__name__)

class RAG(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.initialize_rag()

    def initialize_rag(self):
        """Initialize the RAG components."""
        try:
            # Load documents from the LFG support site
            loader = WebBaseLoader("https://support.lfg.inc/")
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store and retriever
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(splits, embeddings)
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
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise

    @commands.command(name='ask')
    async def ask_question(self, ctx, *, question: str):
        """Ask a question about LFG services."""
        if not self.qa_chain:
            await ctx.send("The question-answering system is not ready yet. Please try again later.")
            return

        try:
            # Show typing indicator
            async with ctx.typing():
                # Get response from QA chain
                result = self.qa_chain({"query": question})
                
                # Format the response
                response = f"**Q:** {question}\n\n**A:** {result['result']}"
                
                # Add sources if available
                if 'source_documents' in result and result['source_documents']:
                    sources = set(doc.metadata.get('source', 'Unknown') for doc in result['source_documents'])
                    sources_text = "\n\n*Sources:*\n" + "\n".join(f"- {src}" for src in sources)
                    response += sources_text
                
                # Send the response
                await ctx.send(response[:2000])  # Discord has a 2000 character limit per message
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            await ctx.send("Sorry, I encountered an error while processing your question. Please try again later.")

    @commands.command(name='refresh')
    @commands.has_permissions(administrator=True)
    async def refresh_knowledge(self, ctx):
        """Refresh the knowledge base (Admin only)."""
        try:
            await ctx.send("Refreshing knowledge base. This may take a moment...")
            self.initialize_rag()
            await ctx.send("Knowledge base refreshed successfully!")
        except Exception as e:
            logger.error(f"Error refreshing knowledge base: {e}")
            await ctx.send("Failed to refresh knowledge base. Check logs for details.")

async def setup(bot):
    await bot.add_cog(RAG(bot))
