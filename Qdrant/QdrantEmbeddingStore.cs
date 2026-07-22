using System.Globalization;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using QdrantRange = Qdrant.Client.Grpc.Range;

namespace LMKit.Data.Storage.Qdrant
{
    /// <summary>
    /// Implements the <see cref="IVectorStore"/> interface using Qdrant as the backend.
    /// Provides operations for creating, deleting, updating, and querying vector data with associated metadata,
    /// leveraging Qdrant's vector search capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Metadata values are stored as string payload fields (the <see cref="MetadataCollection"/> contract).
    /// To support the typed comparisons of <see cref="MetadataFilter"/> (numeric and date ranges, boolean
    /// equality) with Qdrant's native range and match conditions, the store additionally writes typed
    /// shadow payload fields for every metadata value that parses as a number, an ISO 8601 date, or a
    /// boolean: <c>lmkit_typed_num_&lt;key&gt;</c> (double), <c>lmkit_typed_ts_&lt;key&gt;</c> (Unix epoch
    /// milliseconds, double), and <c>lmkit_typed_bool_&lt;key&gt;</c> (bool). Shadow fields are maintained
    /// on upsert and metadata update, hidden from metadata read-back, and used transparently by filter
    /// translation.
    /// </para>
    /// <para>
    /// Documented limits: typed (number, date, boolean) comparisons only match points written by a store
    /// version that maintains shadow fields; points written earlier match string comparisons only. Ordinal
    /// text range comparisons (<c>gt</c>/<c>gte</c>/<c>lt</c>/<c>lte</c> on string values) are not supported
    /// by Qdrant and throw <see cref="NotSupportedException"/>. Date values are shadowed only when stored
    /// in ISO 8601 form (starting with <c>yyyy-MM-dd</c>).
    /// </para>
    /// </remarks>
    public sealed class QdrantEmbeddingStore : IVectorStore, IDisposable
    {
        private const string ShadowPrefix = "lmkit_typed_";
        private const string NumberShadowPrefix = ShadowPrefix + "num_";
        private const string DateShadowPrefix = ShadowPrefix + "ts_";
        private const string BooleanShadowPrefix = ShadowPrefix + "bool_";

        private readonly QdrantClient _client;
        private readonly bool _ownsClient;
        private volatile bool _disposed;

        /// <summary>
        /// Initializes a new instance of the QdrantEmbeddingStore class using the specified QdrantClient.
        /// </summary>
        /// <param name="client">The QdrantClient instance used to communicate with the Qdrant service.</param>
        /// <param name="ownsClient">
        /// If <c>true</c>, the store takes ownership of the client and will dispose it when the store is disposed.
        /// If <c>false</c> (default), the caller retains ownership and is responsible for disposing the client.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="client"/> is <see langword="null"/>.</exception>
        public QdrantEmbeddingStore(QdrantClient client, bool ownsClient = false)
        {
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _ownsClient = ownsClient;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="QdrantEmbeddingStore"/>.
        /// </summary>
        /// <param name="address">The URI of the Qdrant service endpoint. Must not be null.</param>
        /// <param name="apiKey">An optional API key for authentication.</param>
        /// <param name="certificateThumbprint">
        /// An optional SHA-256 certificate thumbprint to enable secure GRPC communication.
        /// If provided, a secure channel is used; otherwise, a standard connection is created.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="address"/> is null.</exception>
        public QdrantEmbeddingStore(Uri address, string apiKey = null, string certificateThumbprint = null)
        {
            if (address == null)
            {
                throw new ArgumentNullException(nameof(address));
            }

            if (!string.IsNullOrWhiteSpace(certificateThumbprint))
            {
                var channel = QdrantChannel.ForAddress(address,
                     new ClientConfiguration
                     {
                         CertificateThumbprint = certificateThumbprint,
                         ApiKey = apiKey
                     }
                   );
                var grpcClient = new QdrantGrpcClient(channel);
                _client = new QdrantClient(grpcClient);
            }
            else
            {
                _client = new QdrantClient(
                    host: address.Host,
                    https: address.Scheme == "https",
                    apiKey: apiKey);
            }

            _ownsClient = true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="QdrantEmbeddingStore"/> class using a pre-configured <see cref="QdrantGrpcClient"/>.
        /// This constructor is intended for use under .NET Framework to support HTTPS connections with secure gRPC communication.
        /// </summary>
        /// <param name="grpcClient">
        /// A pre-configured instance of <see cref="QdrantGrpcClient"/> that is set up to use a secure channel (HTTPS)
        /// with custom certificate thumbprint validation and optional API key authentication.
        /// 
        /// Example usage under .NET Framework:
        /// <code>
        /// // Update with your API key and certificate thumbprint, if any.
        /// string apiKey = ""; // update, if any
        /// string tp = "YOUR_CERTIFICATE_THUMBPRINT";
        /// 
        /// // Create a secure gRPC channel using HTTPS and a custom WinHttpHandler for certificate validation.
        /// var channel = GrpcChannel.ForAddress($"https://localhost:6334", new GrpcChannelOptions
        /// {
        ///     HttpHandler = new WinHttpHandler
        ///     {
        ///         ServerCertificateValidationCallback = CertificateValidation.Thumbprint(tp)
        ///     }
        /// });
        /// 
        /// // Intercept the call to add the API key to metadata.
        /// var callInvoker = channel.Intercept(metadata0 =>
        /// {
        ///     metadata0.Add("api-key", apiKey);
        ///     return metadata0;
        /// });
        /// 
        /// // Create a QdrantGrpcClient using the intercepted call invoker.
        /// var grpcClient = new QdrantGrpcClient(callInvoker);
        /// 
        /// // Instantiate the QdrantEmbeddingStore using the secure gRPC client.
        /// var store = new QdrantEmbeddingStore(grpcClient);
        /// </code>
        /// </param>
        public QdrantEmbeddingStore(QdrantGrpcClient grpcClient)
        {
            _client = new QdrantClient(grpcClient);
            _ownsClient = true;
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
            return await _client.CollectionExistsAsync(collectionIdentifier, cancellationToken: cancellationToken).ConfigureAwait(false);
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

            cancellationToken.ThrowIfCancellationRequested();

            await _client.CreateCollectionAsync(
                collectionIdentifier,
                new VectorParams { Size = vectorSize, Distance = Distance.Cosine },
                cancellationToken: cancellationToken
            ).ConfigureAwait(false);

            if (payloadIndexFields != null)
            {
                foreach (var fieldName in payloadIndexFields)
                {
                    await _client.CreatePayloadIndexAsync(
                        collectionIdentifier,
                        fieldName,
                        PayloadSchemaType.Keyword,
                        cancellationToken: cancellationToken
                    ).ConfigureAwait(false);

                    // Typed shadow fields back the MetadataFilter range and boolean comparisons; index
                    // them alongside the keyword field so typed filters stay indexed on remote servers.
                    await _client.CreatePayloadIndexAsync(
                        collectionIdentifier,
                        NumberShadowPrefix + fieldName,
                        PayloadSchemaType.Float,
                        cancellationToken: cancellationToken
                    ).ConfigureAwait(false);

                    await _client.CreatePayloadIndexAsync(
                        collectionIdentifier,
                        DateShadowPrefix + fieldName,
                        PayloadSchemaType.Float,
                        cancellationToken: cancellationToken
                    ).ConfigureAwait(false);

                    await _client.CreatePayloadIndexAsync(
                        collectionIdentifier,
                        BooleanShadowPrefix + fieldName,
                        PayloadSchemaType.Bool,
                        cancellationToken: cancellationToken
                    ).ConfigureAwait(false);
                }
            }
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
            await _client.DeleteCollectionAsync(collectionIdentifier, cancellationToken: cancellationToken).ConfigureAwait(false);
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
            MetadataCollection metadata = [];
            IReadOnlyList<RetrievedPoint> result;

            if (IsUintId(id))
            {
                result = await _client.RetrieveAsync(collectionIdentifier, ulong.Parse(id), cancellationToken: cancellationToken).ConfigureAwait(false);
            }
            else
            {
                if (!Guid.TryParse(id, out Guid guid))
                {
                    throw new ArgumentException("Invalid GUID format.", nameof(id));
                }

                result = await _client.RetrieveAsync(collectionIdentifier, guid, cancellationToken: cancellationToken).ConfigureAwait(false);
            }

            if (result.Count == 0)
            {
                throw new KeyNotFoundException($"{collectionIdentifier} with id {id} not found");
            }

            foreach (var pair in result[0].Payload)
            {
                if (!IsShadowKey(pair.Key))
                {
                    metadata.Add(PayloadEntryToMetadata(pair));
                }
            }

            return metadata;
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

            return RetrieveCoreAsync(collectionIdentifier, BuildFilterFromMetadata(metadata), options, maxResults, cancellationToken);
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

            return RetrieveCoreAsync(collectionIdentifier, TranslateFilter(filter), options, maxResults, cancellationToken);
        }

        private async Task<List<PointEntry>> RetrieveCoreAsync(
            string collectionIdentifier,
            Filter filter,
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

            var queryResult = await _client.QueryAsync(
                collectionIdentifier,
                filter: filter,
                payloadSelector: new WithPayloadSelector() { Enable = (options & VectorRetrievalOptions.IncludeMetadata) != 0 },
                vectorsSelector: new WithVectorsSelector() { Enable = (options & VectorRetrievalOptions.IncludeVector) != 0 },
                limit: maxResults,
                cancellationToken: cancellationToken
            ).ConfigureAwait(false);

            List<PointEntry> result = new(queryResult.Count);

            foreach (var entry in queryResult)
            {
                result.Add(new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.GetDenseVector().Data, PayloadToMetadata(entry.Payload)));
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
            Filter filter = metadataFilter != null ? BuildFilterFromMetadata(metadataFilter) : null;
            return SearchCoreAsync(collectionIdentifier, vector, limit, options, filter, cancellationToken);
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
            Filter translated = filter != null ? TranslateFilter(filter) : null;
            return SearchCoreAsync(collectionIdentifier, vector, limit, options, translated, cancellationToken);
        }

        private async Task<List<(PointEntry Point, float Score)>> SearchCoreAsync(
            string collectionIdentifier,
            float[] vector,
            uint limit,
            VectorRetrievalOptions options,
            Filter filter,
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

            var queryResult = await _client.SearchAsync(
                collectionIdentifier,
                vector,
                filter: filter,
                payloadSelector: new WithPayloadSelector() { Enable = (options & VectorRetrievalOptions.IncludeMetadata) != 0 },
                vectorsSelector: new WithVectorsSelector() { Enable = (options & VectorRetrievalOptions.IncludeVector) != 0 },
                limit: limit,
                cancellationToken: cancellationToken
            ).ConfigureAwait(false);

            List<(PointEntry Point, float Score)> result = new(queryResult.Count);

            foreach (var entry in queryResult)
            {
                result.Add((new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.GetDenseVector().Data, PayloadToMetadata(entry.Payload)), entry.Score));
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

            return DeleteCoreAsync(collectionIdentifier, BuildFilterFromMetadata(metadata), cancellationToken);
        }

        /// <inheritdoc/>
        public Task DeleteFromMetadataAsync(string collectionIdentifier, MetadataFilter filter, CancellationToken cancellationToken = default)
        {
            if (filter == null)
            {
                throw new ArgumentNullException(nameof(filter));
            }

            return DeleteCoreAsync(collectionIdentifier, TranslateFilter(filter), cancellationToken);
        }

        private async Task DeleteCoreAsync(string collectionIdentifier, Filter filter, CancellationToken cancellationToken)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();

            var updateResult = await _client.DeleteAsync(
                collectionIdentifier,
                filter: filter,
                cancellationToken: cancellationToken
            ).ConfigureAwait(false);

            ThrowIfUpdateFailed(updateResult, $"Failed to delete vector from collection '{collectionIdentifier}'");
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
            var point = new PointStruct
            {
                Id = ParsePointId(id),
                Vectors = vectors
            };

            foreach (var kv in metadata)
            {
                AddPayloadEntryWithShadows(point.Payload, kv.Key, kv.Value);
            }

            var updateResult = await _client.UpsertAsync(collectionIdentifier, [point], cancellationToken: cancellationToken).ConfigureAwait(false);

            ThrowIfUpdateFailed(updateResult, $"Failed to upsert vector for collection '{collectionIdentifier}' with id {id}");
        }

        /// <summary>
        /// Upserts multiple vectors with their associated metadata into the specified collection in a single batch operation.
        /// </summary>
        /// <param name="collectionIdentifier">The name of the collection to upsert vectors into.</param>
        /// <param name="points">A collection of tuples containing the ID, vector data, and metadata for each point.</param>
        /// <param name="cancellationToken">A token to cancel the operation.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <exception cref="ObjectDisposedException">Thrown if the store has been disposed.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="collectionIdentifier"/> is null or empty.</exception>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="points"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown if <paramref name="points"/> is empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the upsert operation fails.</exception>
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

            var pointStructs = new List<PointStruct>();

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

                var point = new PointStruct
                {
                    Id = ParsePointId(id),
                    Vectors = vectors
                };

                foreach (var kv in metadata)
                {
                    AddPayloadEntryWithShadows(point.Payload, kv.Key, kv.Value);
                }

                pointStructs.Add(point);
            }

            if (pointStructs.Count == 0)
            {
                throw new ArgumentException("Points collection cannot be empty.", nameof(points));
            }

            var updateResult = await _client.UpsertAsync(collectionIdentifier, pointStructs, cancellationToken: cancellationToken).ConfigureAwait(false);

            ThrowIfUpdateFailed(updateResult, $"Failed to batch upsert {pointStructs.Count} vectors into collection '{collectionIdentifier}'");
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
            var payload = new Dictionary<string, Value>(metadata.Count);
            foreach (var kv in metadata)
            {
                AddPayloadEntryWithShadows(payload, kv.Key, kv.Value);
            }

            if (mode == MetadataUpdateMode.Replace)
            {
                UpdateResult clearResult = IsUintId(id)
                    ? await _client.ClearPayloadAsync(collectionIdentifier, id: ulong.Parse(id), cancellationToken: cancellationToken).ConfigureAwait(false)
                    : await _client.ClearPayloadAsync(collectionIdentifier, id: new Guid(id), cancellationToken: cancellationToken).ConfigureAwait(false);

                ThrowIfUpdateFailed(clearResult, $"Failed to clear metadata for collection '{collectionIdentifier}' with id {id}");
            }
            else
            {
                // Merge keeps the point's existing payload, so a key whose new value no longer parses in
                // a typed domain must have its stale shadow fields removed, or old typed filters would
                // keep matching the outdated value.
                var staleShadowKeys = new List<string>();
                foreach (var kv in metadata)
                {
                    if (IsShadowKey(kv.Key))
                    {
                        continue;
                    }

                    foreach (string shadowKey in new[] { NumberShadowPrefix + kv.Key, DateShadowPrefix + kv.Key, BooleanShadowPrefix + kv.Key })
                    {
                        if (!payload.ContainsKey(shadowKey))
                        {
                            staleShadowKeys.Add(shadowKey);
                        }
                    }
                }

                if (staleShadowKeys.Count > 0)
                {
                    UpdateResult deleteResult = IsUintId(id)
                        ? await _client.DeletePayloadAsync(collectionIdentifier, staleShadowKeys, id: ulong.Parse(id), cancellationToken: cancellationToken).ConfigureAwait(false)
                        : await _client.DeletePayloadAsync(collectionIdentifier, staleShadowKeys, id: new Guid(id), cancellationToken: cancellationToken).ConfigureAwait(false);

                    ThrowIfUpdateFailed(deleteResult, $"Failed to remove stale typed metadata for collection '{collectionIdentifier}' with id {id}");
                }
            }

            UpdateResult updateResult = IsUintId(id)
                ? await _client.SetPayloadAsync(collectionIdentifier, payload, id: ulong.Parse(id), cancellationToken: cancellationToken).ConfigureAwait(false)
                : await _client.SetPayloadAsync(collectionIdentifier, payload, id: new Guid(id), cancellationToken: cancellationToken).ConfigureAwait(false);

            ThrowIfUpdateFailed(updateResult, $"Failed to update metadata for collection '{collectionIdentifier}' with id {id}");
        }

        /// <inheritdoc/>
        public async Task<List<string>> ListCollectionsAsync(CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();

            var collections = await _client.ListCollectionsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
            var result = new List<string>(collections.Count);

            foreach (var collection in collections)
            {
                result.Add(collection);
            }

            return result;
        }

        /// <summary>
        /// Releases all resources used by the <see cref="QdrantEmbeddingStore"/>.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_ownsClient && _client is IDisposable disposableClient)
            {
                disposableClient.Dispose();
            }

            _disposed = true;
        }

        /// <summary>
        /// Throws an <see cref="ObjectDisposedException"/> if the store has been disposed.
        /// </summary>
        /// <exception cref="ObjectDisposedException">Thrown if the store has been disposed.</exception>
        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(QdrantEmbeddingStore));
            }
        }

        /// <summary>
        /// Validates the result of an update operation and throws an exception if it failed.
        /// </summary>
        /// <param name="result">The update result to validate.</param>
        /// <param name="errorMessage">The error message to include in the exception if validation fails.</param>
        /// <exception cref="InvalidOperationException">Thrown when the update operation did not complete or was not acknowledged.</exception>
        private static void ThrowIfUpdateFailed(UpdateResult result, string errorMessage)
        {
            if (result.Status != UpdateStatus.Completed && result.Status != UpdateStatus.Acknowledged)
            {
                throw new InvalidOperationException($"{errorMessage}. Status: {result.Status}");
            }
        }

        /// <summary>
        /// Builds a Qdrant filter from the provided metadata collection.
        /// </summary>
        /// <param name="metadata">The metadata to convert into filter conditions.</param>
        /// <returns>A <see cref="Filter"/> containing must-match conditions for each metadata entry.</returns>
        private static Filter BuildFilterFromMetadata(MetadataCollection metadata)
        {
            var filter = new Filter();

            foreach (var pair in metadata)
            {
                var condition = new Condition
                {
                    Field = new FieldCondition
                    {
                        Key = pair.Key,
                        Match = new Match { Keyword = pair.Value }
                    }
                };

                filter.Must.Add(condition);
            }

            return filter;
        }

        /// <summary>
        /// Determines whether a payload key is a typed shadow field maintained by this store
        /// (never surfaced as metadata).
        /// </summary>
        private static bool IsShadowKey(string key)
        {
            return key != null && key.StartsWith(ShadowPrefix, StringComparison.Ordinal);
        }

        /// <summary>
        /// Adds a metadata entry to a payload as a string field, plus the typed shadow fields for every
        /// domain the value parses in: a double under <c>lmkit_typed_num_&lt;key&gt;</c>, Unix epoch
        /// milliseconds under <c>lmkit_typed_ts_&lt;key&gt;</c> (ISO 8601 values only), and a bool under
        /// <c>lmkit_typed_bool_&lt;key&gt;</c>. The shadows are what Qdrant's native range and boolean
        /// conditions can filter on, since the primary field is a string.
        /// </summary>
        private static void AddPayloadEntryWithShadows(IDictionary<string, Value> payload, string key, string value)
        {
            payload[key] = new Value { StringValue = value };

            if (IsShadowKey(key) || string.IsNullOrEmpty(value))
            {
                return;
            }

            string trimmed = value.Trim();

            if (double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) &&
                !double.IsNaN(number) && !double.IsInfinity(number))
            {
                payload[NumberShadowPrefix + key] = new Value { DoubleValue = number };
            }

            if (bool.TryParse(trimmed, out bool boolean))
            {
                payload[BooleanShadowPrefix + key] = new Value { BoolValue = boolean };
            }

            if (LooksLikeIsoDate(trimmed) &&
                DateTimeOffset.TryParse(trimmed, CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.AssumeUniversal, out DateTimeOffset date))
            {
                payload[DateShadowPrefix + key] = new Value { DoubleValue = date.ToUnixTimeMilliseconds() };
            }
        }

        /// <summary>
        /// Cheap shape check that gates date shadowing to ISO 8601 values (<c>yyyy-MM-dd...</c>), so
        /// free-form text that a lenient date parser would accept is not misread as a timestamp.
        /// </summary>
        private static bool LooksLikeIsoDate(string value)
        {
            if (value.Length < 10 || value[4] != '-' || value[7] != '-')
            {
                return false;
            }

            for (int i = 0; i < 10; i++)
            {
                if (i == 4 || i == 7)
                {
                    continue;
                }

                if (value[i] < '0' || value[i] > '9')
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Converts a point's payload to a <see cref="MetadataCollection"/>, hiding the typed shadow fields.
        /// </summary>
        private static MetadataCollection PayloadToMetadata(IEnumerable<KeyValuePair<string, Value>> payload)
        {
            MetadataCollection metadata = [];

            if (payload != null)
            {
                foreach (var pair in payload)
                {
                    if (!IsShadowKey(pair.Key))
                    {
                        metadata.Add(PayloadEntryToMetadata(pair));
                    }
                }
            }

            return metadata;
        }

        /// <summary>
        /// Translates a <see cref="MetadataFilter"/> tree into a native Qdrant <see cref="Filter"/>.
        /// String comparisons match the primary string field; numeric, date, and boolean comparisons
        /// target the typed shadow fields, so they only match points written with shadow maintenance.
        /// Every comparison except <c>exists(false)</c> requires the (typed) field to be present, so a
        /// missing key, or a value that does not parse in the requested domain, never matches, mirroring
        /// <see cref="MetadataFilter.Matches"/>.
        /// </summary>
        /// <exception cref="NotSupportedException">
        /// Thrown for ordinal text range comparisons (gt/gte/lt/lte on string values), which Qdrant
        /// cannot evaluate.
        /// </exception>
        private static Filter TranslateFilter(MetadataFilter filter)
        {
            return new Filter { Must = { TranslateNode(filter) } };
        }

        private static Condition TranslateNode(MetadataFilter node)
        {
            if (node is MetadataFilter.Composite composite)
            {
                var inner = new Filter();
                var target = composite.Operator == MetadataFilterOperator.And ? inner.Must : inner.Should;
                foreach (MetadataFilter child in composite.Filters)
                {
                    target.Add(TranslateNode(child));
                }

                return new Condition { Filter = inner };
            }

            var comparison = (MetadataFilter.Comparison)node;
            string key = comparison.Key;

            if (comparison.Operator == MetadataFilterOperator.Exists)
            {
                return comparison.Values[0].BooleanValue
                    ? new Condition { Filter = new Filter { MustNot = { EmptyCondition(key) } } }
                    : EmptyCondition(key);
            }

            switch (comparison.Values[0].Kind)
            {
                case MetadataFilterValueKind.Number:
                    return TranslateRangeComparison(NumberShadowPrefix + key, comparison, value => value.NumberValue);

                case MetadataFilterValueKind.Date:
                    return TranslateRangeComparison(DateShadowPrefix + key, comparison, value => value.DateValue.ToUnixTimeMilliseconds());

                case MetadataFilterValueKind.Boolean:
                    return TranslateBooleanComparison(BooleanShadowPrefix + key, comparison);

                default:
                    return TranslateStringComparison(key, comparison);
            }
        }

        private static Condition TranslateStringComparison(string key, MetadataFilter.Comparison comparison)
        {
            switch (comparison.Operator)
            {
                case MetadataFilterOperator.Equal:
                    return MatchKeyword(key, comparison.Values[0].StringValue);

                case MetadataFilterOperator.NotEqual:
                    return PresentAndNoneOf(key, MatchKeyword(key, comparison.Values[0].StringValue));

                case MetadataFilterOperator.In:
                    return MatchAnyKeyword(key, comparison.Values);

                case MetadataFilterOperator.NotIn:
                    return PresentAndNoneOf(key, MatchAnyKeyword(key, comparison.Values));

                default:
                    throw new NotSupportedException(
                        "Ordinal text range comparisons (gt, gte, lt, lte on string values) are not supported by the " +
                        "Qdrant connector. Use a numeric or date-typed filter value, or restructure the filter.");
            }
        }

        private static Condition TranslateRangeComparison(
            string shadowKey, MetadataFilter.Comparison comparison, Func<MetadataFilterValue, double> convert)
        {
            switch (comparison.Operator)
            {
                case MetadataFilterOperator.Equal:
                    return RangeCondition(shadowKey, new QdrantRange { Gte = convert(comparison.Values[0]), Lte = convert(comparison.Values[0]) });

                case MetadataFilterOperator.NotEqual:
                    return PresentAndNoneOf(shadowKey,
                        RangeCondition(shadowKey, new QdrantRange { Gte = convert(comparison.Values[0]), Lte = convert(comparison.Values[0]) }));

                case MetadataFilterOperator.GreaterThan:
                    return RangeCondition(shadowKey, new QdrantRange { Gt = convert(comparison.Values[0]) });

                case MetadataFilterOperator.GreaterThanOrEqual:
                    return RangeCondition(shadowKey, new QdrantRange { Gte = convert(comparison.Values[0]) });

                case MetadataFilterOperator.LessThan:
                    return RangeCondition(shadowKey, new QdrantRange { Lt = convert(comparison.Values[0]) });

                case MetadataFilterOperator.LessThanOrEqual:
                    return RangeCondition(shadowKey, new QdrantRange { Lte = convert(comparison.Values[0]) });

                case MetadataFilterOperator.In:
                {
                    var any = new Filter();
                    foreach (MetadataFilterValue value in comparison.Values)
                    {
                        any.Should.Add(RangeCondition(shadowKey, new QdrantRange { Gte = convert(value), Lte = convert(value) }));
                    }

                    return new Condition { Filter = any };
                }

                default: // NotIn
                {
                    var matches = new Condition[comparison.Values.Count];
                    for (int i = 0; i < comparison.Values.Count; i++)
                    {
                        matches[i] = RangeCondition(shadowKey, new QdrantRange { Gte = convert(comparison.Values[i]), Lte = convert(comparison.Values[i]) });
                    }

                    return PresentAndNoneOf(shadowKey, matches);
                }
            }
        }

        private static Condition TranslateBooleanComparison(string shadowKey, MetadataFilter.Comparison comparison)
        {
            switch (comparison.Operator)
            {
                case MetadataFilterOperator.Equal:
                    return MatchBoolean(shadowKey, comparison.Values[0].BooleanValue);

                case MetadataFilterOperator.NotEqual:
                    return PresentAndNoneOf(shadowKey, MatchBoolean(shadowKey, comparison.Values[0].BooleanValue));

                case MetadataFilterOperator.In:
                {
                    var any = new Filter();
                    foreach (MetadataFilterValue value in comparison.Values)
                    {
                        any.Should.Add(MatchBoolean(shadowKey, value.BooleanValue));
                    }

                    return new Condition { Filter = any };
                }

                default: // NotIn (range operators over booleans are rejected at filter construction)
                {
                    var matches = new Condition[comparison.Values.Count];
                    for (int i = 0; i < comparison.Values.Count; i++)
                    {
                        matches[i] = MatchBoolean(shadowKey, comparison.Values[i].BooleanValue);
                    }

                    return PresentAndNoneOf(shadowKey, matches);
                }
            }
        }

        private static Condition EmptyCondition(string key)
        {
            return new Condition { IsEmpty = new IsEmptyCondition { Key = key } };
        }

        private static Condition MatchKeyword(string key, string value)
        {
            return new Condition { Field = new FieldCondition { Key = key, Match = new Match { Keyword = value } } };
        }

        private static Condition MatchAnyKeyword(string key, IReadOnlyList<MetadataFilterValue> values)
        {
            var keywords = new RepeatedStrings();
            foreach (MetadataFilterValue value in values)
            {
                keywords.Strings.Add(value.StringValue);
            }

            return new Condition { Field = new FieldCondition { Key = key, Match = new Match { Keywords = keywords } } };
        }

        private static Condition MatchBoolean(string key, bool value)
        {
            return new Condition { Field = new FieldCondition { Key = key, Match = new Match { Boolean = value } } };
        }

        private static Condition RangeCondition(string key, QdrantRange range)
        {
            return new Condition { Field = new FieldCondition { Key = key, Range = range } };
        }

        /// <summary>
        /// Builds the "key is present and none of the given conditions match" filter used for negative
        /// comparisons (<c>ne</c>, <c>nin</c>): the field must exist (a missing key never matches, even
        /// negatively) and every listed positive match is excluded.
        /// </summary>
        private static Condition PresentAndNoneOf(string key, params Condition[] conditions)
        {
            var filter = new Filter { MustNot = { EmptyCondition(key) } };
            foreach (Condition condition in conditions)
            {
                filter.MustNot.Add(condition);
            }

            return new Condition { Filter = filter };
        }

        /// <summary>
        /// Converts a Qdrant payload entry (key-value pair) to a <see cref="Metadata"/> instance.
        /// </summary>
        /// <param name="pair">A key-value pair from the Qdrant payload.</param>
        /// <returns>A new <see cref="Metadata"/> instance representing the key and its corresponding value.</returns>
        private static Metadata PayloadEntryToMetadata(KeyValuePair<string, Value> pair)
        {
            if (pair.Value.HasStringValue)
            {
                return new Metadata(pair.Key, pair.Value.StringValue);
            }

            if (pair.Value.HasDoubleValue)
            {
                return new Metadata(pair.Key, pair.Value.DoubleValue.ToString());
            }

            if (pair.Value.HasBoolValue)
            {
                return new Metadata(pair.Key, pair.Value.BoolValue.ToString());
            }

            if (pair.Value.HasIntegerValue)
            {
                return new Metadata(pair.Key, pair.Value.IntegerValue.ToString());
            }

            if (pair.Value.HasNullValue)
            {
                return new Metadata(pair.Key, pair.Value.NullValue.ToString());
            }

            return new Metadata(pair.Key, pair.Value.ToString());
        }

        /// <summary>
        /// Determines whether the given string identifier represents a numeric (unsigned long) ID.
        /// </summary>
        /// <param name="id">The identifier to test.</param>
        /// <returns><c>true</c> if the identifier can be parsed as an unsigned long; otherwise, <c>false</c>.</returns>
        private static bool IsUintId(string id)
        {
            return ulong.TryParse(id, out _);
        }

        /// <summary>
        /// Parses the provided string identifier into a <see cref="PointId"/>.
        /// </summary>
        /// <param name="id">The identifier to parse.</param>
        /// <returns>A <see cref="PointId"/> corresponding to the provided identifier.</returns>
        /// <exception cref="ArgumentException">Thrown if the identifier is not a valid unsigned long or GUID.</exception>
        private static PointId ParsePointId(string id)
        {
            if (ulong.TryParse(id, out ulong uintId))
            {
                return new PointId(uintId);
            }

            if (Guid.TryParse(id, out Guid guid))
            {
                return new PointId(guid);
            }

            throw new ArgumentException("The provided id is neither a valid unsigned long nor a GUID.", nameof(id));
        }

        /// <summary>
        /// Converts a <see cref="PointId"/> to its string representation.
        /// </summary>
        /// <param name="id">The <see cref="PointId"/> to convert.</param>
        /// <returns>A string representation of the <see cref="PointId"/>.</returns>
        private static string PointIdToString(PointId id)
        {
            return id.HasUuid ? id.Uuid.ToString() : id.Num.ToString();
        }
    }
}