using System.Globalization;
using System.Text;
using Npgsql;
using NpgsqlTypes;

namespace LMKit.Data.Storage.PgVector
{
    /// <summary>
    /// Implements the <see cref="IVectorStore"/> interface using PostgreSQL with the
    /// <see href="https://github.com/pgvector/pgvector">pgvector</see> extension as the backend.
    /// Provides operations for creating, deleting, updating, and querying vector data with associated metadata,
    /// leveraging pgvector's similarity search capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each LM-Kit collection is mapped to a PostgreSQL table inside the configured schema. Every table has three
    /// columns: <c>id</c> (a <c>text</c> primary key), <c>embedding</c> (a <c>vector(N)</c> column whose dimension
    /// is fixed when the collection is created), and <c>metadata</c> (a <c>jsonb</c> column holding the key-value
    /// metadata as a flat JSON object).
    /// </para>
    /// <para>
    /// Similarity is computed with the cosine distance operator (<c>&lt;=&gt;</c>). Scores returned by
    /// <see cref="SearchSimilarVectorsAsync(string, float[], uint, VectorRetrievalOptions, MetadataCollection, CancellationToken)"/>
    /// are cosine similarities (<c>1 - cosine_distance</c>), so a higher
    /// score means a closer match, consistent with the other <see cref="IVectorStore"/> implementations.
    /// </para>
    /// <para>
    /// The store is thread-safe: every operation borrows a pooled connection from an <see cref="NpgsqlDataSource"/>,
    /// so concurrent calls (for example parallel upserts) are supported.
    /// </para>
    /// </remarks>
    public sealed class PgVectorEmbeddingStore : IVectorStore, IDisposable
    {
        private const string DefaultSchema = "public";
        private const int BatchChunkSize = 1000;

        private readonly NpgsqlDataSource _dataSource;
        private readonly bool _ownsDataSource;
        private readonly string _schema;
        private volatile bool _disposed;
        private volatile bool _typedCastHelpersEnsured;
        private readonly SemaphoreSlim _typedCastHelpersGate = new SemaphoreSlim(1, 1);

        /// <summary>
        /// Initializes a new instance of the <see cref="PgVectorEmbeddingStore"/> class using the specified
        /// <see cref="NpgsqlDataSource"/>.
        /// </summary>
        /// <param name="dataSource">
        /// The <see cref="NpgsqlDataSource"/> used to obtain pooled connections to the PostgreSQL service.
        /// The data source must target a database where the <c>vector</c> extension is available.
        /// </param>
        /// <param name="ownsDataSource">
        /// If <c>true</c>, the store takes ownership of the data source and will dispose it when the store is disposed.
        /// If <c>false</c> (default), the caller retains ownership and is responsible for disposing the data source.
        /// </param>
        /// <param name="schema">
        /// The PostgreSQL schema that holds the collection tables. Defaults to <c>public</c> when null or empty.
        /// The schema is created on demand when a collection is created.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="dataSource"/> is <see langword="null"/>.</exception>
        public PgVectorEmbeddingStore(NpgsqlDataSource dataSource, bool ownsDataSource = false, string schema = DefaultSchema)
        {
            _dataSource = dataSource ?? throw new ArgumentNullException(nameof(dataSource));
            _ownsDataSource = ownsDataSource;
            _schema = string.IsNullOrWhiteSpace(schema) ? DefaultSchema : schema;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PgVectorEmbeddingStore"/> class from a PostgreSQL
        /// connection string. The store builds and owns an internal pooled <see cref="NpgsqlDataSource"/>.
        /// </summary>
        /// <param name="connectionString">
        /// A standard Npgsql connection string pointing at the target PostgreSQL database, for example
        /// <c>Host=localhost;Username=postgres;Password=secret;Database=vectors</c>.
        /// </param>
        /// <param name="schema">
        /// The PostgreSQL schema that holds the collection tables. Defaults to <c>public</c> when null or empty.
        /// The schema is created on demand when a collection is created.
        /// </param>
        /// <exception cref="ArgumentException">Thrown when <paramref name="connectionString"/> is null, empty, or whitespace.</exception>
        public PgVectorEmbeddingStore(string connectionString, string schema = DefaultSchema)
        {
            if (string.IsNullOrWhiteSpace(connectionString))
            {
                throw new ArgumentException("Connection string cannot be null or empty.", nameof(connectionString));
            }

            _dataSource = NpgsqlDataSource.Create(connectionString);
            _ownsDataSource = true;
            _schema = string.IsNullOrWhiteSpace(schema) ? DefaultSchema : schema;
        }

        /// <summary>
        /// Ensures that the PostgreSQL database named in the supplied connection string exists, creating it if it
        /// does not. Because a connection cannot be opened to a database that does not yet exist, this method
        /// connects to an existing maintenance database (<paramref name="maintenanceDatabase"/>, <c>postgres</c> by
        /// default) using the same host and credentials, and issues <c>CREATE DATABASE</c> there.
        /// </summary>
        /// <param name="connectionString">
        /// A connection string whose <c>Database</c> value identifies the database to ensure exists. The same host,
        /// port, and credentials are reused to reach the maintenance database.
        /// </param>
        /// <param name="maintenanceDatabase">
        /// An existing database to connect to in order to issue <c>CREATE DATABASE</c>. Defaults to <c>postgres</c>.
        /// </param>
        /// <param name="cancellationToken">An optional token to cancel the operation.</param>
        /// <returns>A task that completes once the target database exists.</returns>
        /// <remarks>
        /// The connecting role must have the <c>CREATEDB</c> privilege and access to the maintenance database.
        /// This method only creates the database; it does not install the <c>vector</c> extension (that is a
        /// server-side installation) nor create any collections.
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown when the connection string is null/empty or specifies no database.</exception>
        public static async Task EnsureDatabaseExistsAsync(string connectionString, string maintenanceDatabase = "postgres", CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(connectionString))
            {
                throw new ArgumentException("Connection string cannot be null or empty.", nameof(connectionString));
            }

            var builder = new NpgsqlConnectionStringBuilder(connectionString);
            string targetDatabase = builder.Database;

            if (string.IsNullOrWhiteSpace(targetDatabase))
            {
                throw new ArgumentException("The connection string must specify a 'Database'.", nameof(connectionString));
            }

            // The target database may not exist yet, so connect to a maintenance database instead.
            builder.Database = string.IsNullOrWhiteSpace(maintenanceDatabase) ? "postgres" : maintenanceDatabase;

            using var connection = new NpgsqlConnection(builder.ConnectionString);
            await connection.OpenAsync(cancellationToken).ConfigureAwait(false);

            using (var existsCommand = new NpgsqlCommand("SELECT 1 FROM pg_database WHERE datname = @name", connection))
            {
                existsCommand.Parameters.AddWithValue("name", targetDatabase);
                var found = await existsCommand.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
                if (found != null)
                {
                    return;
                }
            }

            // CREATE DATABASE cannot run inside a transaction and cannot be parameterized; the identifier is quoted.
            using var createCommand = new NpgsqlCommand($"CREATE DATABASE {QuoteIdentifier(targetDatabase)}", connection);
            try
            {
                await createCommand.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (PostgresException exception) when (exception.SqlState == "42P04")
            {
                // 42P04 = duplicate_database: another caller created it concurrently. Safe to treat as success.
            }
        }

        /// <inheritdoc/>
        public async Task<bool> CollectionExistsAsync(string collectionIdentifier, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables " +
                "WHERE table_schema = @schema AND table_name = @name AND table_type = 'BASE TABLE')",
                connection);
            command.Parameters.AddWithValue("schema", _schema);
            command.Parameters.AddWithValue("name", collectionIdentifier);

            var result = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            return result is bool exists && exists;
        }

        /// <inheritdoc/>
        public async Task CreateCollectionAsync(
            string collectionIdentifier,
            uint vectorSize,
            IEnumerable<string> payloadIndexFields = null,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (vectorSize == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(vectorSize), "Vector size must be greater than zero.");
            }

            cancellationToken.ThrowIfCancellationRequested();

            string tableRef = TableReference(collectionIdentifier);

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var transaction = await connection.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

            using (var command = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", connection, transaction))
            {
                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            using (var command = new NpgsqlCommand($"CREATE SCHEMA IF NOT EXISTS {QuoteIdentifier(_schema)}", connection, transaction))
            {
                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            string createTable =
                $"CREATE TABLE {tableRef} (" +
                "id text PRIMARY KEY, " +
                $"embedding vector({vectorSize.ToString(CultureInfo.InvariantCulture)}) NOT NULL, " +
                "metadata jsonb NOT NULL DEFAULT '{}'::jsonb)";

            using (var command = new NpgsqlCommand(createTable, connection, transaction))
            {
                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            if (payloadIndexFields != null)
            {
                foreach (var fieldName in payloadIndexFields)
                {
                    if (string.IsNullOrWhiteSpace(fieldName))
                    {
                        continue;
                    }

                    string indexName = IndexName(collectionIdentifier, fieldName);
                    string createIndex =
                        $"CREATE INDEX IF NOT EXISTS {QuoteIdentifier(indexName)} ON {tableRef} " +
                        $"((metadata ->> '{EscapeStringLiteral(fieldName)}'))";

                    using var command = new NpgsqlCommand(createIndex, connection, transaction);
                    await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
                }
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public async Task DeleteCollectionAsync(string collectionIdentifier, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand($"DROP TABLE IF EXISTS {TableReference(collectionIdentifier)}", connection);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public async Task<MetadataCollection> GetMetadataAsync(string collectionIdentifier, string id, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (string.IsNullOrWhiteSpace(id))
            {
                throw new ArgumentException("ID cannot be null or empty.", nameof(id));
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand($"SELECT metadata FROM {TableReference(collectionIdentifier)} WHERE id = @id", connection);
            command.Parameters.AddWithValue("id", id);

            using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);

            if (!await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                throw new KeyNotFoundException($"{collectionIdentifier} with id {id} not found");
            }

            return reader.IsDBNull(0)
                ? new MetadataCollection()
                : MetadataCollection.FromJson(reader.GetString(0));
        }

        /// <inheritdoc/>
        public Task<List<PointEntry>> RetrieveFromMetadataAsync(
            string collectionIdentifier,
            MetadataCollection metadata,
            VectorRetrievalOptions options,
            uint maxResults,
            CancellationToken cancellationToken = default)
        {
            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            return RetrieveCoreAsync(
                collectionIdentifier,
                command => new TranslatedFilter { Conditions = AppendMetadataConditions(metadata, command) },
                options, maxResults, cancellationToken);
        }

        /// <inheritdoc/>
        public Task<List<PointEntry>> RetrieveFromMetadataAsync(
            string collectionIdentifier,
            MetadataFilter filter,
            VectorRetrievalOptions options,
            uint maxResults,
            CancellationToken cancellationToken = default)
        {
            if (filter == null)
            {
                throw new ArgumentNullException(nameof(filter));
            }

            return RetrieveCoreAsync(
                collectionIdentifier,
                command => TranslateFilter(filter, command),
                options, maxResults, cancellationToken);
        }

        private async Task<List<PointEntry>> RetrieveCoreAsync(
            string collectionIdentifier,
            Func<NpgsqlCommand, TranslatedFilter> filterBuilder,
            VectorRetrievalOptions options,
            uint maxResults,
            CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (maxResults == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxResults), "Max results must be greater than zero.");
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand { Connection = connection };

            TranslatedFilter filter = filterBuilder(command);
            if (filter.NeedsTypedCastHelpers)
            {
                await EnsureTypedCastHelpersAsync(connection, cancellationToken).ConfigureAwait(false);
            }

            bool getVector = (options & VectorRetrievalOptions.IncludeVector) != 0;
            bool getMetadata = (options & VectorRetrievalOptions.IncludeMetadata) != 0;

            var query = new StringBuilder("SELECT id");
            int ordinal = 1;
            int vectorOrdinal = -1;
            int metadataOrdinal = -1;

            if (getVector)
            {
                query.Append(", embedding::text");
                vectorOrdinal = ordinal++;
            }

            if (getMetadata)
            {
                query.Append(", metadata");
                metadataOrdinal = ordinal++;
            }

            query.Append(" FROM ").Append(TableReference(collectionIdentifier));

            if (filter.Conditions.Length > 0)
            {
                query.Append(" WHERE ").Append(filter.Conditions);
            }

            query.Append(" LIMIT @limit");
            command.Parameters.AddWithValue("limit", (long)maxResults);
            command.CommandText = query.ToString();

            var result = new List<PointEntry>();

            using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                result.Add(ReadPointEntry(reader, vectorOrdinal, metadataOrdinal));
            }

            return result;
        }

        /// <inheritdoc/>
        public Task<List<(PointEntry Point, float Score)>> SearchSimilarVectorsAsync(
            string collectionIdentifier,
            float[] vector,
            uint limit,
            VectorRetrievalOptions options,
            MetadataCollection metadataFilter,
            CancellationToken cancellationToken = default)
        {
            return SearchCoreAsync(
                collectionIdentifier, vector, limit, options,
                command => metadataFilter != null
                    ? new TranslatedFilter { Conditions = AppendMetadataConditions(metadataFilter, command) }
                    : new TranslatedFilter(),
                cancellationToken);
        }

        /// <inheritdoc/>
        public Task<List<(PointEntry Point, float Score)>> SearchSimilarVectorsAsync(
            string collectionIdentifier,
            float[] vector,
            uint limit,
            VectorRetrievalOptions options,
            MetadataFilter filter,
            CancellationToken cancellationToken = default)
        {
            return SearchCoreAsync(
                collectionIdentifier, vector, limit, options,
                command => filter != null ? TranslateFilter(filter, command) : new TranslatedFilter(),
                cancellationToken);
        }

        private async Task<List<(PointEntry Point, float Score)>> SearchCoreAsync(
            string collectionIdentifier,
            float[] vector,
            uint limit,
            VectorRetrievalOptions options,
            Func<NpgsqlCommand, TranslatedFilter> filterBuilder,
            CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (vector == null || vector.Length == 0)
            {
                throw new ArgumentException("Vector cannot be null or empty.", nameof(vector));
            }

            if (limit == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(limit), "Limit must be greater than zero.");
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand { Connection = connection };
            command.Parameters.AddWithValue("query", FormatVector(vector));

            TranslatedFilter filter = filterBuilder(command);
            if (filter.NeedsTypedCastHelpers)
            {
                await EnsureTypedCastHelpersAsync(connection, cancellationToken).ConfigureAwait(false);
            }

            bool getVector = (options & VectorRetrievalOptions.IncludeVector) != 0;
            bool getMetadata = (options & VectorRetrievalOptions.IncludeMetadata) != 0;

            var query = new StringBuilder("SELECT id");
            int ordinal = 1;
            int vectorOrdinal = -1;
            int metadataOrdinal = -1;

            if (getVector)
            {
                query.Append(", embedding::text");
                vectorOrdinal = ordinal++;
            }

            if (getMetadata)
            {
                query.Append(", metadata");
                metadataOrdinal = ordinal++;
            }

            int scoreOrdinal = ordinal;
            query.Append(", 1 - (embedding <=> @query::vector) AS score");
            query.Append(" FROM ").Append(TableReference(collectionIdentifier));

            if (filter.Conditions.Length > 0)
            {
                query.Append(" WHERE ").Append(filter.Conditions);
            }

            query.Append(" ORDER BY embedding <=> @query::vector LIMIT @limit");
            command.Parameters.AddWithValue("limit", (long)limit);
            command.CommandText = query.ToString();

            var result = new List<(PointEntry Point, float Score)>();

            using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                var point = ReadPointEntry(reader, vectorOrdinal, metadataOrdinal);
                float score = reader.IsDBNull(scoreOrdinal) ? 0f : (float)reader.GetDouble(scoreOrdinal);
                result.Add((point, score));
            }

            return result;
        }

        /// <inheritdoc/>
        public Task DeleteFromMetadataAsync(string collectionIdentifier, MetadataCollection metadata, CancellationToken cancellationToken = default)
        {
            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            return DeleteCoreAsync(
                collectionIdentifier,
                command => new TranslatedFilter { Conditions = AppendMetadataConditions(metadata, command) },
                cancellationToken);
        }

        /// <inheritdoc/>
        public Task DeleteFromMetadataAsync(string collectionIdentifier, MetadataFilter filter, CancellationToken cancellationToken = default)
        {
            if (filter == null)
            {
                throw new ArgumentNullException(nameof(filter));
            }

            return DeleteCoreAsync(
                collectionIdentifier,
                command => TranslateFilter(filter, command),
                cancellationToken);
        }

        private async Task DeleteCoreAsync(
            string collectionIdentifier,
            Func<NpgsqlCommand, TranslatedFilter> filterBuilder,
            CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand { Connection = connection };

            TranslatedFilter filter = filterBuilder(command);
            if (filter.NeedsTypedCastHelpers)
            {
                await EnsureTypedCastHelpersAsync(connection, cancellationToken).ConfigureAwait(false);
            }

            command.CommandText = filter.Conditions.Length > 0
                ? $"DELETE FROM {TableReference(collectionIdentifier)} WHERE {filter.Conditions}"
                : $"DELETE FROM {TableReference(collectionIdentifier)}";

            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public async Task UpsertAsync(string collectionIdentifier, string id, float[] vectors, MetadataCollection metadata, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (string.IsNullOrWhiteSpace(id))
            {
                throw new ArgumentException("ID cannot be null or empty.", nameof(id));
            }

            if (vectors == null || vectors.Length == 0)
            {
                throw new ArgumentException("Vector data cannot be null or empty.", nameof(vectors));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            cancellationToken.ThrowIfCancellationRequested();

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand(
                $"INSERT INTO {TableReference(collectionIdentifier)} (id, embedding, metadata) " +
                "VALUES (@id, @embedding::vector, @metadata) " +
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                connection);

            command.Parameters.AddWithValue("id", id);
            command.Parameters.AddWithValue("embedding", FormatVector(vectors));
            command.Parameters.Add(new NpgsqlParameter("metadata", NpgsqlDbType.Jsonb) { Value = metadata.ToJson() });

            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Upserts multiple vectors with their associated metadata into the specified collection in a single batched
        /// transaction. The points are written in chunks, and either all chunks succeed or the transaction is rolled back.
        /// </summary>
        /// <param name="collectionIdentifier">The name of the collection to upsert vectors into.</param>
        /// <param name="points">A collection of tuples containing the ID, vector data, and metadata for each point.</param>
        /// <param name="cancellationToken">A token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if the store has been disposed.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="collectionIdentifier"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="points"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="points"/> is empty or contains an invalid entry.</exception>
        public async Task UpsertBatchAsync(
            string collectionIdentifier,
            IEnumerable<(string Id, float[] Vectors, MetadataCollection Metadata)> points,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (points == null)
            {
                throw new ArgumentNullException(nameof(points));
            }

            cancellationToken.ThrowIfCancellationRequested();

            var rows = new List<(string Id, string Vector, string Metadata)>();

            foreach (var (id, vectors, metadata) in points)
            {
                if (string.IsNullOrWhiteSpace(id))
                {
                    throw new ArgumentException("Point ID cannot be null or empty.", nameof(points));
                }

                if (vectors == null || vectors.Length == 0)
                {
                    throw new ArgumentException($"Vector data cannot be null or empty for point with id '{id}'.", nameof(points));
                }

                if (metadata == null)
                {
                    throw new ArgumentNullException(nameof(points), $"Metadata cannot be null for point with id '{id}'.");
                }

                rows.Add((id, FormatVector(vectors), metadata.ToJson()));
            }

            if (rows.Count == 0)
            {
                throw new ArgumentException("Points collection cannot be empty.", nameof(points));
            }

            string tableRef = TableReference(collectionIdentifier);

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var transaction = await connection.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

            for (int offset = 0; offset < rows.Count; offset += BatchChunkSize)
            {
                int count = Math.Min(BatchChunkSize, rows.Count - offset);

                var query = new StringBuilder($"INSERT INTO {tableRef} (id, embedding, metadata) VALUES ");
                using var command = new NpgsqlCommand { Connection = connection, Transaction = transaction };

                for (int i = 0; i < count; i++)
                {
                    var (id, vector, metadata) = rows[offset + i];
                    string suffix = i.ToString(CultureInfo.InvariantCulture);

                    if (i > 0)
                    {
                        query.Append(", ");
                    }

                    query.Append("(@id").Append(suffix)
                         .Append(", @e").Append(suffix).Append("::vector, @m").Append(suffix).Append(')');

                    command.Parameters.AddWithValue("id" + suffix, id);
                    command.Parameters.AddWithValue("e" + suffix, vector);
                    command.Parameters.Add(new NpgsqlParameter("m" + suffix, NpgsqlDbType.Jsonb) { Value = metadata });
                }

                query.Append(" ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata");
                command.CommandText = query.ToString();

                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
        }

        /// <inheritdoc/>
        public async Task UpdateMetadataAsync(string collectionIdentifier, string id, MetadataCollection metadata, MetadataUpdateMode mode, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (string.IsNullOrWhiteSpace(id))
            {
                throw new ArgumentException("ID cannot be null or empty.", nameof(id));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            cancellationToken.ThrowIfCancellationRequested();

            // Replace swaps the metadata column for the new object; Merge concatenates the new entries into the
            // existing object, with the new values overriding existing keys (the jsonb concatenation operator).
            string assignment = mode == MetadataUpdateMode.Replace ? "@metadata" : "metadata || @metadata";

            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand(
                $"UPDATE {TableReference(collectionIdentifier)} SET metadata = {assignment} WHERE id = @id",
                connection);

            command.Parameters.AddWithValue("id", id);
            command.Parameters.Add(new NpgsqlParameter("metadata", NpgsqlDbType.Jsonb) { Value = metadata.ToJson() });

            int affected = await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

            if (affected == 0)
            {
                throw new KeyNotFoundException($"{collectionIdentifier} with id {id} not found");
            }
        }

        /// <inheritdoc/>
        public async Task<List<string>> ListCollectionsAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();

            // A collection is any base table in the configured schema that exposes an 'embedding' column of type vector.
            using var connection = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var command = new NpgsqlCommand(
                "SELECT c.relname FROM pg_class c " +
                "JOIN pg_namespace n ON n.oid = c.relnamespace " +
                "JOIN pg_attribute a ON a.attrelid = c.oid " +
                "JOIN pg_type t ON t.oid = a.atttypid " +
                "WHERE n.nspname = @schema AND c.relkind = 'r' AND a.attname = 'embedding' AND t.typname = 'vector' " +
                "ORDER BY c.relname",
                connection);
            command.Parameters.AddWithValue("schema", _schema);

            var result = new List<string>();

            using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                result.Add(reader.GetString(0));
            }

            return result;
        }

        /// <summary>
        /// Releases all resources used by the <see cref="PgVectorEmbeddingStore"/>.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_ownsDataSource)
            {
                _dataSource.Dispose();
            }

            _typedCastHelpersGate.Dispose();
            _disposed = true;
        }

        private Task<NpgsqlConnection> OpenConnectionAsync(CancellationToken cancellationToken)
        {
            return _dataSource.OpenConnectionAsync(cancellationToken).AsTask();
        }

        /// <summary>
        /// Throws an <see cref="ObjectDisposedException"/> if the store has been disposed.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown if the store has been disposed.</exception>
        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(PgVectorEmbeddingStore));
            }
        }

        /// <summary>
        /// Reads a <see cref="PointEntry"/> from the current row of a reader produced by a retrieve or search query.
        /// </summary>
        /// <param name="reader">The data reader positioned on the row to read.</param>
        /// <param name="vectorOrdinal">The column ordinal of the vector text, or a negative value when the vector was not selected.</param>
        /// <param name="metadataOrdinal">The column ordinal of the metadata json, or a negative value when the metadata was not selected.</param>
        /// <returns>A populated <see cref="PointEntry"/>.</returns>
        private static PointEntry ReadPointEntry(NpgsqlDataReader reader, int vectorOrdinal, int metadataOrdinal)
        {
            string id = reader.GetString(0);

            float[] vector = null;
            if (vectorOrdinal >= 0 && !reader.IsDBNull(vectorOrdinal))
            {
                vector = ParseVector(reader.GetString(vectorOrdinal));
            }

            MetadataCollection metadata = metadataOrdinal >= 0 && !reader.IsDBNull(metadataOrdinal)
                ? MetadataCollection.FromJson(reader.GetString(metadataOrdinal))
                : new MetadataCollection();

            return new PointEntry(id, vector, metadata);
        }

        /// <summary>
        /// Appends a metadata equality filter to the supplied command, one condition per metadata entry, and returns
        /// the SQL fragment (without the leading <c>WHERE</c>). Parameter values are bound to the command.
        /// </summary>
        /// <param name="metadata">The metadata to convert into filter conditions. May be empty.</param>
        /// <param name="command">The command to which parameter values are added.</param>
        /// <returns>A conjunction of conditions, or an empty string when no metadata entries are supplied.</returns>
        private static string AppendMetadataConditions(MetadataCollection metadata, NpgsqlCommand command)
        {
            if (metadata == null || metadata.Count == 0)
            {
                return string.Empty;
            }

            var builder = new StringBuilder();
            int index = 0;

            foreach (var pair in metadata)
            {
                if (index > 0)
                {
                    builder.Append(" AND ");
                }

                string parameterName = "mf" + index.ToString(CultureInfo.InvariantCulture);
                builder.Append("metadata ->> '")
                       .Append(EscapeStringLiteral(pair.Key))
                       .Append("' = @")
                       .Append(parameterName);

                command.Parameters.AddWithValue(parameterName, pair.Value);
                index++;
            }

            return builder.ToString();
        }

        /// <summary>
        /// A metadata filter rendered as a SQL fragment, with the flag indicating whether the fragment
        /// references the typed-cast helper functions (which must exist before the query runs).
        /// </summary>
        private sealed class TranslatedFilter
        {
            public string Conditions = string.Empty;
            public bool NeedsTypedCastHelpers;
            public int ParameterIndex;
        }

        /// <summary>
        /// Renders a <see cref="MetadataFilter"/> tree as a parameterized SQL condition over the
        /// <c>metadata</c> jsonb column, mirroring the in-memory <see cref="MetadataFilter.Matches"/>
        /// semantics: text compares ordinally (byte order via <c>COLLATE "C"</c>), numbers and ISO 8601
        /// dates go through exception-safe cast helpers so a stored value that does not parse in the
        /// requested domain never matches (and never fails the query), and every comparison except
        /// <c>exists(false)</c> requires the key to be present.
        /// </summary>
        /// <param name="filter">The filter tree to translate.</param>
        /// <param name="command">The command to which parameter values are bound.</param>
        /// <returns>The translated filter fragment.</returns>
        private TranslatedFilter TranslateFilter(MetadataFilter filter, NpgsqlCommand command)
        {
            var result = new TranslatedFilter();
            result.Conditions = TranslateNode(filter, command, result);
            return result;
        }

        private string TranslateNode(MetadataFilter node, NpgsqlCommand command, TranslatedFilter state)
        {
            if (node is MetadataFilter.Composite composite)
            {
                var parts = new List<string>(composite.Filters.Count);
                foreach (MetadataFilter child in composite.Filters)
                {
                    parts.Add(TranslateNode(child, command, state));
                }

                string separator = composite.Operator == MetadataFilterOperator.And ? " AND " : " OR ";
                return "(" + string.Join(separator, parts) + ")";
            }

            var comparison = (MetadataFilter.Comparison)node;
            string keyLiteral = "'" + EscapeStringLiteral(comparison.Key) + "'";
            string accessor = "metadata ->> " + keyLiteral;

            if (comparison.Operator == MetadataFilterOperator.Exists)
            {
                return comparison.Values[0].BooleanValue
                    ? "(metadata ? " + keyLiteral + ")"
                    : "(NOT (metadata ? " + keyLiteral + "))";
            }

            // The stored value, interpreted in the filter value's domain. An uninterpretable stored
            // value (or a missing key) yields SQL NULL, and every comparison below propagates NULL,
            // which the surrounding WHERE / NOT treats as "no match".
            MetadataFilterValueKind kind = comparison.Values[0].Kind;
            string stored;
            switch (kind)
            {
                case MetadataFilterValueKind.Number:
                    state.NeedsTypedCastHelpers = true;
                    stored = TypedCastFunctionReference("lmkit_try_numeric") + "(" + accessor + ")::float8";
                    break;
                case MetadataFilterValueKind.Date:
                    state.NeedsTypedCastHelpers = true;
                    stored = TypedCastFunctionReference("lmkit_try_timestamptz") + "(" + accessor + ")";
                    break;
                case MetadataFilterValueKind.Boolean:
                    stored = "(CASE WHEN lower(btrim(" + accessor + ")) IN ('true','false') THEN btrim(" + accessor + ")::boolean END)";
                    break;
                default:
                    bool ordinalRange = comparison.Operator is MetadataFilterOperator.GreaterThan
                        or MetadataFilterOperator.GreaterThanOrEqual
                        or MetadataFilterOperator.LessThan
                        or MetadataFilterOperator.LessThanOrEqual;
                    stored = ordinalRange ? "((" + accessor + ") COLLATE \"C\")" : "(" + accessor + ")";
                    break;
            }

            if (comparison.Operator is MetadataFilterOperator.In or MetadataFilterOperator.NotIn)
            {
                string arrayParameter = BindArrayParameter(comparison.Values, kind, command, state);
                return comparison.Operator == MetadataFilterOperator.In
                    ? "(" + stored + " = ANY(" + arrayParameter + "))"
                    : "(NOT (" + stored + " = ANY(" + arrayParameter + ")))";
            }

            string parameter = BindScalarParameter(comparison.Values[0], command, state);
            string sqlOperator = comparison.Operator switch
            {
                MetadataFilterOperator.Equal => "=",
                MetadataFilterOperator.NotEqual => "<>",
                MetadataFilterOperator.GreaterThan => ">",
                MetadataFilterOperator.GreaterThanOrEqual => ">=",
                MetadataFilterOperator.LessThan => "<",
                _ => "<="
            };

            return "(" + stored + " " + sqlOperator + " " + parameter + ")";
        }

        private static string BindScalarParameter(MetadataFilterValue value, NpgsqlCommand command, TranslatedFilter state)
        {
            string name = "mfx" + state.ParameterIndex.ToString(CultureInfo.InvariantCulture);
            state.ParameterIndex++;
            command.Parameters.AddWithValue(name, BoxFilterValue(value));
            return "@" + name;
        }

        private static string BindArrayParameter(
            IReadOnlyList<MetadataFilterValue> values, MetadataFilterValueKind kind, NpgsqlCommand command, TranslatedFilter state)
        {
            string name = "mfx" + state.ParameterIndex.ToString(CultureInfo.InvariantCulture);
            state.ParameterIndex++;

            object array;
            switch (kind)
            {
                case MetadataFilterValueKind.Number:
                    var numbers = new double[values.Count];
                    for (int i = 0; i < values.Count; i++) { numbers[i] = values[i].NumberValue; }
                    array = numbers;
                    break;
                case MetadataFilterValueKind.Date:
                    var dates = new DateTime[values.Count];
                    for (int i = 0; i < values.Count; i++) { dates[i] = values[i].DateValue.UtcDateTime; }
                    array = dates;
                    break;
                case MetadataFilterValueKind.Boolean:
                    var booleans = new bool[values.Count];
                    for (int i = 0; i < values.Count; i++) { booleans[i] = values[i].BooleanValue; }
                    array = booleans;
                    break;
                default:
                    var strings = new string[values.Count];
                    for (int i = 0; i < values.Count; i++) { strings[i] = values[i].StringValue; }
                    array = strings;
                    break;
            }

            command.Parameters.AddWithValue(name, array);
            return "@" + name;
        }

        private static object BoxFilterValue(MetadataFilterValue value)
        {
            switch (value.Kind)
            {
                case MetadataFilterValueKind.Number:
                    return value.NumberValue;
                case MetadataFilterValueKind.Boolean:
                    return value.BooleanValue;
                case MetadataFilterValueKind.Date:
                    // Npgsql maps a UTC DateTime to timestamptz without kind ambiguity.
                    return value.DateValue.UtcDateTime;
                default:
                    return value.StringValue;
            }
        }

        /// <summary>
        /// Builds the schema-qualified, safely-quoted reference to a typed-cast helper function.
        /// </summary>
        private string TypedCastFunctionReference(string functionName)
        {
            return QuoteIdentifier(_schema) + "." + QuoteIdentifier(functionName);
        }

        /// <summary>
        /// Creates the exception-safe cast helper functions used by typed filter comparisons, once per
        /// store instance. <c>lmkit_try_numeric</c> and <c>lmkit_try_timestamptz</c> return NULL for a
        /// value that does not parse, so malformed metadata never fails a filtered query; the timestamp
        /// helper pins the timezone to UTC so values without an explicit offset are read as UTC,
        /// matching the in-memory evaluation.
        /// </summary>
        private async Task EnsureTypedCastHelpersAsync(NpgsqlConnection connection, CancellationToken cancellationToken)
        {
            if (_typedCastHelpersEnsured)
            {
                return;
            }

            await _typedCastHelpersGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_typedCastHelpersEnsured)
                {
                    return;
                }

                string sql =
                    $"CREATE SCHEMA IF NOT EXISTS {QuoteIdentifier(_schema)}; " +
                    $"CREATE OR REPLACE FUNCTION {TypedCastFunctionReference("lmkit_try_numeric")}(p_value text) " +
                    "RETURNS numeric LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$ " +
                    "BEGIN RETURN btrim(p_value)::numeric; EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$; " +
                    $"CREATE OR REPLACE FUNCTION {TypedCastFunctionReference("lmkit_try_timestamptz")}(p_value text) " +
                    "RETURNS timestamptz LANGUAGE plpgsql STABLE PARALLEL SAFE SET timezone = 'UTC' AS $$ " +
                    "BEGIN RETURN btrim(p_value)::timestamptz; EXCEPTION WHEN OTHERS THEN RETURN NULL; END $$";

                using var command = new NpgsqlCommand(sql, connection);
                await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
                _typedCastHelpersEnsured = true;
            }
            finally
            {
                _typedCastHelpersGate.Release();
            }
        }

        /// <summary>
        /// Formats a vector as a pgvector text literal of the form <c>[v0,v1,...]</c> using a round-trippable,
        /// culture-invariant representation for each component.
        /// </summary>
        /// <param name="vector">The vector to format.</param>
        /// <returns>The pgvector text representation of the vector.</returns>
        private static string FormatVector(float[] vector)
        {
            var builder = new StringBuilder(vector.Length * 8 + 2);
            builder.Append('[');

            for (int i = 0; i < vector.Length; i++)
            {
                if (i > 0)
                {
                    builder.Append(',');
                }

                builder.Append(vector[i].ToString("G9", CultureInfo.InvariantCulture));
            }

            builder.Append(']');
            return builder.ToString();
        }

        /// <summary>
        /// Parses a pgvector text literal of the form <c>[v0,v1,...]</c> into a float array.
        /// </summary>
        /// <param name="text">The pgvector text representation to parse.</param>
        /// <returns>The parsed vector, or an empty array when the input is empty.</returns>
        private static float[] ParseVector(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return Array.Empty<float>();
            }

            string body = text.Trim();

            if (body.Length >= 2 && body[0] == '[' && body[body.Length - 1] == ']')
            {
                body = body.Substring(1, body.Length - 2);
            }

            if (body.Length == 0)
            {
                return Array.Empty<float>();
            }

            string[] parts = body.Split(',');
            var result = new float[parts.Length];

            for (int i = 0; i < parts.Length; i++)
            {
                result[i] = float.Parse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture);
            }

            return result;
        }

        /// <summary>
        /// Builds the fully-qualified, safely-quoted table reference (<c>"schema"."collection"</c>) for a collection.
        /// </summary>
        /// <param name="collectionIdentifier">The collection identifier.</param>
        /// <returns>The quoted, schema-qualified table reference.</returns>
        private string TableReference(string collectionIdentifier)
        {
            return QuoteIdentifier(_schema) + "." + QuoteIdentifier(collectionIdentifier);
        }

        /// <summary>
        /// Builds a deterministic, length-bounded index name for a payload field of a collection.
        /// </summary>
        /// <param name="collectionIdentifier">The collection identifier.</param>
        /// <param name="fieldName">The metadata field being indexed.</param>
        /// <returns>An index name guaranteed to fit within PostgreSQL's identifier length limit.</returns>
        private string IndexName(string collectionIdentifier, string fieldName)
        {
            uint hash = Fnv1a($"{_schema}.{collectionIdentifier}.{fieldName}");
            return "lmkv_idx_" + hash.ToString("x8", CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// Computes a 32-bit FNV-1a hash of the supplied string. Used to derive short, stable index names.
        /// </summary>
        /// <param name="value">The string to hash.</param>
        /// <returns>The 32-bit FNV-1a hash.</returns>
        private static uint Fnv1a(string value)
        {
            const uint offsetBasis = 2166136261;
            const uint prime = 16777619;

            uint hash = offsetBasis;
            foreach (char c in value)
            {
                hash ^= c;
                hash *= prime;
            }

            return hash;
        }

        /// <summary>
        /// Quotes a PostgreSQL identifier, escaping embedded double quotes so it is safe to interpolate into SQL.
        /// </summary>
        /// <param name="identifier">The identifier to quote.</param>
        /// <returns>The double-quoted identifier.</returns>
        private static string QuoteIdentifier(string identifier)
        {
            return "\"" + identifier.Replace("\"", "\"\"") + "\"";
        }

        /// <summary>
        /// Escapes a single-quoted SQL string literal by doubling embedded single quotes.
        /// </summary>
        /// <param name="value">The literal value to escape.</param>
        /// <returns>The escaped value, safe to place inside single quotes.</returns>
        private static string EscapeStringLiteral(string value)
        {
            return value.Replace("'", "''");
        }
    }
}
