package de.kherud.llama;

import java.nio.ByteBuffer;

/**
 * Container that allows slicing an {@link ByteBuffer}
 * with arbitrary slicing lengths on Java versions older than 13.
 * Does not extend ByteBuffer because the super constructor
 * requires memory segment proxies, and we can't access the delegate's.
 * Does not implement Buffer because the {@link java.nio.Buffer#slice(int, int)}
 * method is specifically blocked from being implemented or used on older jdk versions.
 * Unfortunately this can't be generified with {@link SliceableIntBuffer}, since there is no shared interface.
 */
class SliceableByteBuffer {
	final ByteBuffer delegate;

	private final int offset;

	private final int capacity;

	SliceableByteBuffer(ByteBuffer delegate) {
		this.delegate = delegate;
		this.capacity = delegate.capacity();
		this.offset = 0;
	}

	SliceableByteBuffer(ByteBuffer delegate, int offset, int capacity) {
		this.delegate = delegate;
		this.offset = offset;
		this.capacity = capacity;
	}

	SliceableByteBuffer slice(int offset, int length) {
		// Where the magic happens
		// Wrapping is equivalent to the slice operation so long
		// as you keep track of your offsets and capacities.
		// So, we use this container class to track those offsets and translate
		// them to the correct frame of reference.
		return new SliceableByteBuffer(
				ByteBuffer.wrap(
						this.delegate.array(),
						this.offset + offset,
						length
				),
				this.offset + offset,
				length
		);

	}

	int capacity() {
		return capacity;
	}

	SliceableByteBuffer put(int index, byte b) {
		delegate.put(offset + index, b);
		return this;
	}

	byte get(int index) {
		return delegate.get(offset + index);
	}

	void clear() {
		// Clear set the limit and position
		// to 0 and capacity respectively,
		// but that's not what the buffer was initially
		// after the wrap() call, so we manually
		// set the limit and position to what they were
		// after the wrap call.
		delegate.clear();
		delegate.limit(offset + capacity);
		delegate.position(offset);
	}

}
