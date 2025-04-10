using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace LMKit.Data.Storage.Qdrant
{
    /// <summary>
    /// Implements the <see cref="IVectorStore"/> interface using Qdrant as the backend.
    /// Provides operations for creating, deleting, updating, and querying vector data with associated metadata,
    /// leveraging Qdrant's vector search capabilities.
    /// </summary>
    public sealed class QdrantEmbeddingStore : IVectorStore
    {
        private readonly QdrantClient _client;


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
                _client = new QdrantClient(address ?? throw new ArgumentNullException(nameof(address)), apiKey);
            }
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
        }

        /// <inheritdoc/>
        public async Task<bool> CollectionExistsAsync(string collectionIdentifier, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();
            return await _client.CollectionExistsAsync(collectionIdentifier, cancellationToken: cancellationToken);
        }

        /// <inheritdoc/>
        public async Task CreateCollectionAsync(string collectionIdentifier, uint vectorSize, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();
            await _client.CreateCollectionAsync(
                collectionIdentifier,
                new VectorParams { Size = vectorSize, Distance = Distance.Cosine },
                cancellationToken: cancellationToken
            );
        }

        /// <inheritdoc/>
        public async Task DeleteCollectionAsync(string collectionIdentifier, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            cancellationToken.ThrowIfCancellationRequested();
            await _client.DeleteCollectionAsync(collectionIdentifier, cancellationToken: cancellationToken);
        }

        /// <inheritdoc/>
        public async Task<MetadataCollection> GetMetadataAsync(string collectionIdentifier, string id, CancellationToken cancellationToken = default)
        {
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
                result = await _client.RetrieveAsync(collectionIdentifier, ulong.Parse(id), cancellationToken: cancellationToken);
            }
            else
            {
                if (!Guid.TryParse(id, out Guid guid))
                {
                    throw new ArgumentException("Invalid GUID format.", nameof(id));
                }

                result = await _client.RetrieveAsync(collectionIdentifier, guid, cancellationToken: cancellationToken);
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
        public async Task<List<PointEntry>> RetrieveFromMetadataAsync(string collectionIdentifier, MetadataCollection metadata, bool getVector, bool getMetadata, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            cancellationToken.ThrowIfCancellationRequested();
            var filter = new Filter();

            foreach (var pair in metadata)
            {
                var condition = new Condition
                {
                    Field = new FieldCondition
                    {
                        Key = pair.Key,
                        Match = new Match { Text = pair.Value }
                    }
                };

                filter.Must.Add(condition);
            }

            var queryResult = await _client.QueryAsync(
                collectionIdentifier,
                filter: filter,
                payloadSelector: new WithPayloadSelector() { Enable = getMetadata },
                vectorsSelector: new WithVectorsSelector() { Enable = getVector },
                limit: ulong.MaxValue,
                cancellationToken: cancellationToken
            );

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
                result.Add(new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.Data, metadataResponse));
            }

            return result;
        }

        /// <inheritdoc/>
        public async Task<List<(PointEntry Point, float Score)>> SearchSimilarVectorsAsync(string collectionIdentifier, float[] vector, uint limit, bool getVector, bool getMetadata, CancellationToken cancellationToken = default)
        {
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
                payloadSelector: new WithPayloadSelector() { Enable = getMetadata },
                vectorsSelector: new WithVectorsSelector() { Enable = getVector },
                limit: limit,
                cancellationToken: cancellationToken
            );

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
                result.Add((new PointEntry(PointIdToString(entry.Id), entry.Vectors?.Vector?.Data, metadataResponse), entry.Score));
            }

            return result;
        }

        /// <inheritdoc/>
        public async Task DeleteFromMetadataAsync(string collectionIdentifier, MetadataCollection metadata, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(collectionIdentifier))
            {
                throw new ArgumentException("Collection identifier cannot be null or empty.", nameof(collectionIdentifier));
            }

            if (metadata == null)
            {
                throw new ArgumentNullException(nameof(metadata));
            }

            cancellationToken.ThrowIfCancellationRequested();
            var filter = new Filter();

            foreach (var pair in metadata)
            {
                var condition = new Condition
                {
                    Field = new FieldCondition
                    {
                        Key = pair.Key,
                        Match = new Match { Text = pair.Value }
                    }
                };

                filter.Must.Add(condition);
            }

            var updateResult = await _client.DeleteAsync(
                collectionIdentifier,
                filter: filter,
                cancellationToken: cancellationToken
            );

            if (updateResult.Status != UpdateStatus.Completed)
            {
                throw new Exception($"Failed to delete vector from collection '{collectionIdentifier}'");
            }
        }

        /// <inheritdoc/>
        public async Task UpsertAsync(string collectionIdentifier, string id, float[] vectors, MetadataCollection metadata, CancellationToken cancellationToken = default)
        {
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
                point.Payload.Add(kv.Key, kv.Value);
            }

            var updateResult = await _client.UpsertAsync(collectionIdentifier, new[] { point }, cancellationToken: cancellationToken);

            if (updateResult.Status != UpdateStatus.Completed)
            {
                throw new Exception($"Failed to upsert vector for collection '{collectionIdentifier}' with id {id}");
            }
        }

        /// <inheritdoc/>
        public async Task UpdateMetadataAsync(string collectionIdentifier, string id, MetadataCollection metadata, bool clearFirst, CancellationToken cancellationToken = default)
        {
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
            // Build payload dictionary using metadata key-value pairs.
            var payload = new Dictionary<string, Value>(metadata.Count);
            foreach (var kv in metadata)
            {
                payload.Add(kv.Key, new Value { StringValue = kv.Value });
            }

            if (clearFirst)
            {
                UpdateResult clearResult = IsUintId(id)
                    ? await _client.ClearPayloadAsync(collectionIdentifier, id: ulong.Parse(id), cancellationToken: cancellationToken)
                    : await _client.ClearPayloadAsync(collectionIdentifier, id: new Guid(id), cancellationToken: cancellationToken);

                if (clearResult.Status != UpdateStatus.Completed)
                {
                    throw new Exception($"Failed to clear metadata for collection '{collectionIdentifier}' with id {id}");
                }
            }

            UpdateResult updateResult = IsUintId(id)
                ? await _client.SetPayloadAsync(collectionIdentifier, payload, id: ulong.Parse(id), cancellationToken: cancellationToken)
                : await _client.SetPayloadAsync(collectionIdentifier, payload, id: new Guid(id), cancellationToken: cancellationToken);

            if (updateResult.Status != UpdateStatus.Completed)
            {
                throw new Exception($"Failed to update metadata for collection '{collectionIdentifier}' with id {id}");
            }
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