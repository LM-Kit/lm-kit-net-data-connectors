using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace LMKit.Data.Storage.Qdrant
{
    /// <summary>
    /// Implements the <see cref="IVectorStore"/> interface using Qdrant as the backend.
    /// Provides operations for creating, deleting, updating, and querying vector data with associated metadata,
    /// leveraging Qdrant's vector search capabilities.
    /// </summary>
    public sealed class QdrantEmbeddingStore : IVectorStore, IDisposable
    {
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
                metadata.Add(PayloadEntryToMetadata(pair));
            }

            return metadata;
        }

        /// <inheritdoc/>
        public async Task<List<PointEntry>> RetrieveFromMetadataAsync(
            string collectionIdentifier,
            MetadataCollection metadata,
            VectorRetrievalOptions options,
            uint maxResults,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            if (maxResults == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(maxResults), "Max results must be greater than zero.");
            }

            cancellationToken.ThrowIfCancellationRequested();
            var filter = BuildFilterFromMetadata(metadata);

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
                MetadataCollection metadataResponse = [];
                if (entry.Payload != null)
                {
                    foreach (var pair in entry.Payload)
                    {
                        metadataResponse.Add(PayloadEntryToMetadata(pair));
                    }
                }
                result.Add(new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.GetDenseVector().Data, metadataResponse));
            }

            return result;
        }

        /// <inheritdoc/>
        public async Task<List<(PointEntry Point, float Score)>> SearchSimilarVectorsAsync(
            string collectionIdentifier,
            float[] vector,
            uint limit,
            VectorRetrievalOptions options,
            MetadataCollection metadataFilter,
            CancellationToken cancellationToken = default)
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

            Filter filter = metadataFilter != null ? BuildFilterFromMetadata(metadataFilter) : null;

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
                MetadataCollection metadataResponse = [];
                if (entry.Payload != null)
                {
                    foreach (var pair in entry.Payload)
                    {
                        metadataResponse.Add(PayloadEntryToMetadata(pair));
                    }
                }
                result.Add((new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.GetDenseVector().Data, metadataResponse), entry.Score));
            }

            return result;
        }

        /// <inheritdoc/>
        public async Task DeleteFromMetadataAsync(string collectionIdentifier, MetadataCollection metadata, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            cancellationToken.ThrowIfCancellationRequested();
            var filter = BuildFilterFromMetadata(metadata);

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
                point.Payload.Add(kv.Key, new Value { StringValue = kv.Value });

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
                    point.Payload.Add(kv.Key, new Value { StringValue = kv.Value });
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
                payload.Add(kv.Key, new Value { StringValue = kv.Value });
            }

            if (mode == MetadataUpdateMode.Replace)
            {
                UpdateResult clearResult = IsUintId(id)
                    ? await _client.ClearPayloadAsync(collectionIdentifier, id: ulong.Parse(id), cancellationToken: cancellationToken).ConfigureAwait(false)
                    : await _client.ClearPayloadAsync(collectionIdentifier, id: new Guid(id), cancellationToken: cancellationToken).ConfigureAwait(false);

                ThrowIfUpdateFailed(clearResult, $"Failed to clear metadata for collection '{collectionIdentifier}' with id {id}");
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