"""release 3 sources ingestion versioning artifacts

Revision ID: f3b7c9d2a610
Revises: e2a91c7b5d30
Create Date: 2026-07-09 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "f3b7c9d2a610"
down_revision = "e2a91c7b5d30"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    op.create_table(
        "knowledge_sources",
        sa.Column("source_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("upload_time", sa.String(), nullable=False),
        sa.Column("processing_status", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_deleted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.CheckConstraint("processing_status IN ('pending', 'processing', 'indexed', 'failed', 'deleted')", name="ck_knowledge_sources_processing_status"),
        sa.CheckConstraint("is_deleted IN (0, 1)", name="ck_knowledge_sources_is_deleted"),
    )
    op.create_index("idx_sources_user_status", "knowledge_sources", ["user_id", "processing_status"])
    op.create_index("idx_sources_user_hash", "knowledge_sources", ["user_id", "content_hash"])

    op.add_column("knowledge_chunks", sa.Column("replaces_chunk_id", sa.String(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("replaced_by_chunk_id", sa.String(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("change_reason", sa.Text(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("modified_by_user_query", sa.Text(), nullable=True))
    op.create_index("idx_chunks_user_source", "knowledge_chunks", ["user_id", "source_id"])
    if _is_postgres():
        op.create_foreign_key(
            "fk_knowledge_chunks_source_id",
            "knowledge_chunks",
            "knowledge_sources",
            ["source_id"],
            ["source_id"],
        )

    op.create_table(
        "generated_artifacts",
        sa.Column("artifact_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_hop_id", sa.String(), nullable=True),
        sa.Column("file_type", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("storage_url", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("expires_at", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.CheckConstraint("file_type IN ('xlsx', 'pdf', 'pptx', 'txt', 'csv')", name="ck_generated_artifacts_file_type"),
        sa.CheckConstraint("status IN ('created', 'deleted', 'failed')", name="ck_generated_artifacts_status"),
    )
    op.create_index("idx_artifacts_user_status", "generated_artifacts", ["user_id", "status"])


def downgrade() -> None:
    op.drop_index("idx_artifacts_user_status", table_name="generated_artifacts")
    op.drop_table("generated_artifacts")
    if _is_postgres():
        op.drop_constraint("fk_knowledge_chunks_source_id", "knowledge_chunks", type_="foreignkey")
    op.drop_index("idx_chunks_user_source", table_name="knowledge_chunks")
    op.drop_column("knowledge_chunks", "modified_by_user_query")
    op.drop_column("knowledge_chunks", "change_reason")
    op.drop_column("knowledge_chunks", "replaced_by_chunk_id")
    op.drop_column("knowledge_chunks", "replaces_chunk_id")
    op.drop_index("idx_sources_user_hash", table_name="knowledge_sources")
    op.drop_index("idx_sources_user_status", table_name="knowledge_sources")
    op.drop_table("knowledge_sources")
